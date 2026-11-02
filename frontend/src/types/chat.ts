/**
 * Data contracts and TypeScript type definitions for the PDF RAG Assistant.
 */

export interface SourceItem {
  source: string;
  page_number: number;
  chunk_id: number;
  score: number;
  text: string;
}

export interface DebugTelemetry {
  original_query: string;
  expanded_query: string;
  faiss_candidates_count: number;
  bm25_candidates_count: number;
  hybrid_candidates_count: number;
  cross_encoder_scores?: Array<{
    chunk_id: number;
    score: number;
    page_number: number;
  }>;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  corrected_question?: string | null;
  retrieval_mode?: string;
  sources?: SourceItem[];
  debug_info?: DebugTelemetry | null;
  isError?: boolean;
}

export interface DocumentInfo {
  filename: string;
  file_path: string;
  total_pages: number;
  total_chunks: number;
  chunk_size: number;
  chunk_overlap: number;
  embedding_model: string;
  cross_encoder_model: string;
  llm_model: string;
  retrieval_methods: string[];
  status: string;
}

export interface HealthInfo {
  status: string;
  rag_initialized: boolean;
  ollama_status: string;
  ollama_model: string;
  document_loaded: boolean;
}

export interface ChatApiResponse {
  original_question: string;
  corrected_question?: string | null;
  answer: string;
  retrieval_mode: string;
  sources: SourceItem[];
  debug_info?: DebugTelemetry | null;
}
