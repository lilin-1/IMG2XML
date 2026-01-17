import React, { useState, useEffect, useContext } from 'react';
import axios from 'axios';
import { AuthContext } from './AuthContext';
import { X, Loader2, Save } from 'lucide-react';

const API_BASE = "/api";

export default function SettingsModal({ onClose }) {
  const { auth } = useContext(AuthContext);
  const [loading, setLoading] = useState(false);
  const [keys, setKeys] = useState({
    openai_api_key: '',
    openai_base_url: ''
  });
  
  // Ideally, fetch current settings on mount if the backend supported returning hidden keys (it doesn't, for security).
  // But we can know IF they are set.
  // For now, these are just "Update" fields.

  const handleSave = async () => {
    setLoading(true);
    try {
      await axios.put(
        `${API_BASE}/auth/keys`, 
        {
          openai_api_key: keys.openai_api_key || null,
          openai_base_url: keys.openai_base_url || null
        },
        { headers: { Authorization: `Bearer ${auth.token}` } }
      );
      alert("Settings updated successfully!");
      onClose();
    } catch (err) {
      alert("Failed to update settings: " + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-md mx-4 overflow-hidden">
        <div className="flex items-center justify-between p-4 border-b">
          <h3 className="text-lg font-semibold text-gray-900">User Settings (BYOK)</h3>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-500">
            <X className="w-5 h-5" />
          </button>
        </div>
        
        <div className="p-6 space-y-4">
          <p className="text-sm text-gray-500">
            Bring Your Own Key to bypass server credits limits or use specific models.
            Leave blank to keep existing or use server default.
          </p>
          
          <div>
            <label className="block text-sm font-medium text-gray-700">OpenAI API Key</label>
            <input
              type="password"
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm border p-2"
              placeholder="sk-..."
              value={keys.openai_api_key}
              onChange={e => setKeys({...keys, openai_api_key: e.target.value})}
            />
          </div>

          <div>
             <label className="block text-sm font-medium text-gray-700">Custom Base URL (Optional)</label>
             <input
              type="text"
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm border p-2"
              placeholder="e.g. https://api.openai.com/v1"
              value={keys.openai_base_url}
              onChange={e => setKeys({...keys, openai_base_url: e.target.value})}
             />
          </div>
        </div>

        <div className="flex items-center justify-end p-4 border-t bg-gray-50">
          <button
            onClick={onClose}
            className="px-4 py-2 mr-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={loading}
            className="flex items-center px-4 py-2 text-sm font-medium text-white bg-indigo-600 rounded-md hover:bg-indigo-700 disabled:opacity-50"
          >
            {loading && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
            Save Changes
          </button>
        </div>
      </div>
    </div>
  );
}
