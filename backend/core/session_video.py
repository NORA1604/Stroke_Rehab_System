"""Session evidence video capture.

The mobile client already streams JPEG frames to /ws/pose for realtime
pose scoring. We piggyback on that stream to capture a short evidence clip.

Important low-memory design:
- Only a bounded number of JPEG frames are retained.
- Frames are decoded and sent to the encoder ONE AT A TIME.
- Decoded frames are not accumulated in a Python list.
- Video storage uses ONE background worker, preventing multiple FFmpeg
  jobs from running simultaneously on a small Render instance.
"""

import logging
import os
import queue
import re
import tempfile
import threading
from typing import List, Optional

from core.supabase_db import (
    delete_other_session_video_rows,
    delete_storage_object,
    insert_session_video,
    list_other_session_video_paths,
    session_video_row_exists_for_path,
    upload_to_storage,
)

logger = logging.getLogger("uvicorn.error")


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

_BUCKET = "session-evidence"


# ---------------------------------------------------------------------------
# Per-path locking
# ---------------------------------------------------------------------------

_path_locks: dict = {}
_path_locks_guard = threading.Lock()


def _lock_for_path(path: str) -> threading.Lock:
    with _path_locks_guard:
        lock = _path_locks.get(path)
        if lock is None:
            lock = threading.Lock()
            _path_locks[path] = lock
        return lock


# ---------------------------------------------------------------------------
# Capture limits
# ---------------------------------------------------------------------------

# Keep the evidence clip short.
_MAX_CLIP_SECONDS = 10.0

# Hard memory bound.
#
# Your realtime stream is normally only a few FPS, so 60 frames is enough
# for a 10-second evidence clip while providing a strong memory ceiling.
_MAX_CLIP_FRAMES = 60


# ---------------------------------------------------------------------------
# Video encoding
# ---------------------------------------------------------------------------

_OUTPUT_MAX_HEIGHT = 480

_MIN_FPS = 1.0
_MAX_FPS = 15.0
_FALLBACK_FPS = 6.0


# ---------------------------------------------------------------------------
# Single background video worker
# ---------------------------------------------------------------------------

# IMPORTANT:
# Never create one encoding thread per exercise.
#
# A queue + one worker means only ONE FFmpeg/imageio encoding operation can
# exist at a time.
#
# maxsize=1 prevents completed clips from accumulating indefinitely.
_CLIP_QUEUE: "queue.Queue[Optional[SessionClipRecorder]]" = queue.Queue(maxsize=1)


class SessionClipRecorder:
    """Buffers raw JPEG frames for one exercise's evidence clip."""

    def __init__(
        self,
        patient_id: str,
        session_id: str,
        exercise_type: str,
    ) -> None:
        self.patient_id = (patient_id or "").strip()
        self.session_id = (session_id or "").strip()
        self.exercise_type = (exercise_type or "").strip()

        self.enabled = bool(
            self.patient_id and self.session_id
        )

        self._frames: List[bytes] = []
        self._start: Optional[float] = None
        self._last: Optional[float] = None
        self._done = False

    def add_frame(
        self,
        jpeg_bytes: bytes,
        pose_detected: bool = True,
    ) -> bool:
        """Add a JPEG frame.

        Returns True when the capture window has finished and the caller
        should submit this recorder to the background worker.
        """

        if (
            not self.enabled
            or self._done
            or not jpeg_bytes
        ):
            return False

        now = _now()

        # Don't start the evidence clip until a pose is detected.
        if self._start is None:
            if not pose_detected:
                return False

            self._start = now

        self._last = now

        # Store a bytes object rather than keeping references to mutable
        # buffers supplied by the WebSocket implementation.
        self._frames.append(bytes(jpeg_bytes))

        elapsed = now - self._start

        if (
            len(self._frames) >= _MAX_CLIP_FRAMES
            or elapsed >= _MAX_CLIP_SECONDS
        ):
            self._done = True
            return True

        return False

    def capture_fps(self) -> float:
        """Estimate the real arrival rate of frames."""

        count = len(self._frames)

        if (
            count <= 1
            or self._start is None
            or self._last is None
        ):
            return _FALLBACK_FPS

        elapsed = self._last - self._start

        if elapsed <= 0:
            return _FALLBACK_FPS

        fps = (count - 1) / elapsed

        return max(
            _MIN_FPS,
            min(_MAX_FPS, fps),
        )

    @property
    def done(self) -> bool:
        return self._done

    def has_frames(self) -> bool:
        return bool(self._frames)

    def finalize(self) -> None:
        """Close the capture window without adding another frame."""

        self._done = True

    def frames(self) -> List[bytes]:
        return self._frames


def _now() -> float:
    import time

    return time.monotonic()


def _safe_exercise_slug(exercise_type: str) -> str:
    slug = re.sub(
        r"[^a-z0-9]+",
        "_",
        (exercise_type or "").lower(),
    ).strip("_")

    return slug or "exercise"


# ---------------------------------------------------------------------------
# Low-memory encoder
# ---------------------------------------------------------------------------

def _encode_frames_to_mp4(
    jpeg_frames: List[bytes],
    fps: float,
) -> Optional[bytes]:
    """Encode JPEG frames into an H.264 MP4.

    IMPORTANT:
    Frames are decoded and immediately handed to the encoder.

    We intentionally DO NOT build:

        decoded = [frame1, frame2, frame3, ...]

    because that can consume a large amount of RAM on Render.
    """

    if not jpeg_frames:
        return None

    import cv2
    import numpy as np
    import imageio

    tmp_path: Optional[str] = None
    writer = None

    try:
        # Determine output dimensions from the first valid frame.
        out_w: Optional[int] = None
        out_h: Optional[int] = None

        # We need to find the first decodable frame before creating
        # the encoder.
        first_rgb = None

        for raw in jpeg_frames:
            arr = np.frombuffer(raw, np.uint8)

            img = cv2.imdecode(
                arr,
                cv2.IMREAD_COLOR,
            )

            if img is None:
                continue

            h, w = img.shape[:2]

            if h > _OUTPUT_MAX_HEIGHT:
                scale = _OUTPUT_MAX_HEIGHT / float(h)

                img = cv2.resize(
                    img,
                    (
                        max(1, int(round(w * scale))),
                        _OUTPUT_MAX_HEIGHT,
                    ),
                )

            oh, ow = img.shape[:2]

            out_h = max(2, oh - (oh % 2))
            out_w = max(2, ow - (ow % 2))

            if (
                img.shape[1] != out_w
                or img.shape[0] != out_h
            ):
                img = cv2.resize(
                    img,
                    (out_w, out_h),
                )

            first_rgb = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2RGB,
            )

            break

        if first_rgb is None or out_w is None or out_h is None:
            return None

        with tempfile.NamedTemporaryFile(
            suffix=".mp4",
            delete=False,
        ) as tmp:
            tmp_path = tmp.name

        writer = imageio.get_writer(
            tmp_path,
            format="FFMPEG",
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=1,
            output_params=[
                "-preset",
                "veryfast",
                "-crf",
                "30",
            ],
        )

        # Write the first frame.
        writer.append_data(first_rgb)

        # IMPORTANT:
        # Decode/write one frame at a time.
        for raw in jpeg_frames[1:]:
            arr = np.frombuffer(
                raw,
                np.uint8,
            )

            img = cv2.imdecode(
                arr,
                cv2.IMREAD_COLOR,
            )

            if img is None:
                continue

            h, w = img.shape[:2]

            if h > _OUTPUT_MAX_HEIGHT:
                scale = _OUTPUT_MAX_HEIGHT / float(h)

                img = cv2.resize(
                    img,
                    (
                        max(
                            1,
                            int(round(w * scale)),
                        ),
                        _OUTPUT_MAX_HEIGHT,
                    ),
                )

            if (
                img.shape[1] != out_w
                or img.shape[0] != out_h
            ):
                img = cv2.resize(
                    img,
                    (out_w, out_h),
                )

            rgb = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2RGB,
            )

            writer.append_data(rgb)

            # Explicitly release references before the next frame.
            del rgb
            del img
            del arr

        writer.close()
        writer = None

        # The current upload_to_storage API expects bytes, so the final MP4
        # still needs to be read into memory. The important improvement is
        # that decoded video frames are no longer all held in RAM at once.
        with open(tmp_path, "rb") as handle:
            mp4 = handle.read()

        return mp4

    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass

        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def store_clip(
    recorder: SessionClipRecorder,
) -> None:
    """Encode → upload → index → purge old sessions."""

    try:
        if (
            not recorder.enabled
            or not recorder.has_frames()
        ):
            return

        fps = recorder.capture_fps()

        mp4 = _encode_frames_to_mp4(
            recorder.frames(),
            fps,
        )

        if not mp4:
            logger.warning(
                "session-video: no encodable frames "
                "(patient=%s exercise=%s)",
                recorder.patient_id,
                recorder.exercise_type,
            )
            return

        slug = _safe_exercise_slug(
            recorder.exercise_type
        )

        storage_path = (
            f"{recorder.patient_id}/"
            f"{recorder.session_id}/"
            f"{slug}.mp4"
        )

        with _lock_for_path(storage_path):

            upload = upload_to_storage(
                _BUCKET,
                storage_path,
                mp4,
                "video/mp4",
            )

            if not upload.get("stored"):
                logger.warning(
                    "session-video: upload failed: %s",
                    upload,
                )
                return

            duration = round(
                len(recorder.frames()) / fps,
                1,
            )

            index_payload = {
                "patient_id": recorder.patient_id,
                "session_id": recorder.session_id,
                "exercise_type": recorder.exercise_type,
                "storage_path": storage_path,
                "duration_seconds": duration,
            }

            indexed = insert_session_video(
                index_payload
            )

            if not indexed.get("stored"):
                indexed = insert_session_video(
                    index_payload
                )

            if not indexed.get("stored"):
                row_exists = (
                    session_video_row_exists_for_path(
                        storage_path
                    )
                )

                if row_exists is False:
                    logger.warning(
                        "session-video: index row NOT stored "
                        "after retry; removing orphaned upload: %s",
                        storage_path,
                    )

                    if not delete_storage_object(
                        _BUCKET,
                        storage_path,
                    ):
                        logger.warning(
                            "session-video: failed to delete "
                            "orphaned upload: %s",
                            storage_path,
                        )
                else:
                    logger.warning(
                        "session-video: index row NOT stored "
                        "after retry; leaving upload in place "
                        "(row_exists=%s): %s",
                        row_exists,
                        storage_path,
                    )

                return

        # Only purge after successful indexing.
        _purge_other_sessions(
            recorder.patient_id,
            recorder.session_id,
        )

        logger.info(
            "session-video: stored %s (%d frames, %.1fs)",
            storage_path,
            len(recorder.frames()),
            duration,
        )

    except Exception:
        logger.exception(
            "session-video: store_clip failed"
        )

    finally:
        # Release the large JPEG buffer as soon as possible.
        recorder._frames.clear()


def _purge_other_sessions(
    patient_id: str,
    session_id: str,
) -> None:
    """Keep only the current session's clips."""

    try:
        stale_paths = (
            list_other_session_video_paths(
                patient_id,
                session_id,
            )
        )

        for path in stale_paths:
            delete_storage_object(
                _BUCKET,
                path,
            )

        delete_other_session_video_rows(
            patient_id,
            session_id,
        )

        if stale_paths:
            logger.info(
                "session-video: purged %d old clip(s) "
                "for patient=%s",
                len(stale_paths),
                patient_id,
            )

    except Exception:
        logger.exception(
            "session-video: purge failed"
        )


# ---------------------------------------------------------------------------
# Single worker
# ---------------------------------------------------------------------------

def _clip_worker() -> None:
    """Process evidence clips sequentially."""

    while True:
        try:
            recorder = _CLIP_QUEUE.get()

            if recorder is None:
                return

            try:
                store_clip(recorder)
            except Exception:
                logger.exception(
                    "session-video: worker failure"
                )
            finally:
                _CLIP_QUEUE.task_done()

        except Exception:
            logger.exception(
                "session-video: worker loop failure"
            )


_clip_worker_thread = threading.Thread(
    target=_clip_worker,
    name="session-video-worker",
    daemon=True,
)

_clip_worker_thread.start()


def store_clip_async(
    recorder: SessionClipRecorder,
) -> None:
    """Queue a clip for the single background worker.

    Never creates an additional encoding thread.
    """

    if (
        recorder is None
        or not recorder.enabled
        or not recorder.has_frames()
    ):
        return

    try:
        _CLIP_QUEUE.put_nowait(recorder)

    except queue.Full:
        # We deliberately drop the oldest memory pressure rather than
        # allowing several large video buffers to accumulate and kill
        # the Render process.
        logger.warning(
            "session-video: encoder queue full; "
            "dropping evidence clip "
            "(patient=%s exercise=%s)",
            recorder.patient_id,
            recorder.exercise_type,
        )

        recorder._frames.clear()