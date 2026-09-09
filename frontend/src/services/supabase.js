import { createClient } from '@supabase/supabase-js';
import * as SecureStore from 'expo-secure-store';


// This adapter tells Supabase to securely save the user's login token
// using React Native's native secure storage, rather than web cookies.
const ExpoSecureStoreAdapter = { // Connects Supabase Auth to Expo SecureStore.
  getItem: (key) => { 
    // Retrieves a stored value.
    console.log("[SecureStore] getItem key:", JSON.stringify(key)); 
    // Logs the key for debugging.
    return SecureStore.getItemAsync(key); 
    // Gets the value from SecureStore.
  },
  setItem: (key, value) => { 
    // Saves a value.
    console.log("[SecureStore] setItem key:", JSON.stringify(key)); 
    // Logs the key for debugging.
    return SecureStore.setItemAsync(key, value); 
    // Stores the value securely.
  },
  removeItem: (key) => { 
    // Deletes a stored value.
    console.log("[SecureStore] removeItem key:", JSON.stringify(key)); 
    // Logs the key for debugging.
    return SecureStore.deleteItemAsync(key); 
    // Removes the value from SecureStore.
  },
};


const supabaseUrl = process.env.EXPO_PUBLIC_SUPABASE_URL;
const supabaseAnonKey = process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseAnonKey) {
  console.error("Missing Supabase environment variables:");
  if (!supabaseUrl) console.error("- EXPO_PUBLIC_SUPABASE_URL");
  if (!supabaseAnonKey) console.error("- EXPO_PUBLIC_SUPABASE_ANON_KEY");
  throw new Error("Supabase configuration is incomplete. Please check your .env file.");
}


//sets up how the frontend talks to supabase and stores the users data
export const supabase = createClient(supabaseUrl, supabaseAnonKey, {
  auth: {
    //tells to use the auth manager
    storage: ExpoSecureStoreAdapter,
    //automatically refresh the token so the user stays logged in
    autoRefreshToken: true,
    //tells the app to remember the user
    persistSession: true,
    //tells the app not to detect the session in the url
    detectSessionInUrl: false,
  },
});