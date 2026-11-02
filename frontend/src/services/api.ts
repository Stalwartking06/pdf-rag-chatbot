/**
 * API client service for communicating with the FastAPI RAG backend.
 */

import axios from 'axios';
import type { ChatApiResponse, DocumentInfo, HealthInfo } from '../types/chat';

// In development with Vite proxy, '/api' routes to http://127.0.0.1:8000/api
// If VITE_API_URL is specified, use that base URL.
const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_URL || '',
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 60000, // 60s timeout to allow local LLM generation
});

export const apiService = {
  /**
   * Check backend server, RAG pipeline, and Ollama status.
   */
  async getHealth(): Promise<HealthInfo> {
    const response = await apiClient.get<HealthInfo>('/api/health');
    return response.data;
  },

  /**
   * Fetch real document and retrieval pipeline metadata.
   */
  async getDocumentInfo(): Promise<DocumentInfo> {
    const response = await apiClient.get<DocumentInfo>('/api/document');
    return response.data;
  },

  /**
   * Send user question to the RAG pipeline.
   */
  async sendChatQuestion(question: string): Promise<ChatApiResponse> {
    const response = await apiClient.post<ChatApiResponse>('/api/chat', { question });
    return response.data;
  },

  /**
   * Reset conversation state on the server.
   */
  async resetChat(): Promise<{ status: string; message: string }> {
    const response = await apiClient.post<{ status: string; message: string }>('/api/chat/reset');
    return response.data;
  },
};
