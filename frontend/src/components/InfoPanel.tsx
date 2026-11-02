import React from 'react';
import { 
  X, 
  FileText, 
  Cpu, 
  Database, 
  Server, 
  Info 
} from 'lucide-react';
import type { DocumentInfo, HealthInfo } from '../types/chat';

interface InfoPanelProps {
  documentInfo: DocumentInfo | null;
  healthInfo: HealthInfo | null;
  isOpen: boolean;
  onClose: () => void;
}

export const InfoPanel: React.FC<InfoPanelProps> = ({
  documentInfo,
  healthInfo,
  isOpen,
  onClose,
}) => {
  if (!isOpen) return null;

  return (
    <aside className="right-info-panel glass-panel">
      <div className="info-panel-header">
        <div className="panel-title-group">
          <Info size={18} className="info-icon" />
          <h3>System & Document Info</h3>
        </div>
        <button 
          type="button" 
          className="close-panel-btn"
          onClick={onClose}
          aria-label="Close information panel"
          title="Close panel"
        >
          <X size={16} />
        </button>
      </div>

      <div className="info-panel-content">
        {/* Document Stats Card */}
        <div className="info-card glass-card">
          <div className="info-card-header">
            <FileText size={16} className="card-header-icon" />
            <span>Document Details</span>
          </div>
          <div className="info-stat-list">
            <div className="info-stat-item">
              <span className="info-stat-label">File:</span>
              <span className="info-stat-value code-font">
                {documentInfo?.filename || 'company_policy.pdf'}
              </span>
            </div>
            <div className="info-stat-item">
              <span className="info-stat-label">Total Pages:</span>
              <span className="info-stat-value font-bold">
                {documentInfo?.total_pages || 109} Pages
              </span>
            </div>
            <div className="info-stat-item">
              <span className="info-stat-label">Total Chunks:</span>
              <span className="info-stat-value font-bold">
                {documentInfo?.total_chunks || 285} Chunks
              </span>
            </div>
            <div className="info-stat-item">
              <span className="info-stat-label">Chunk Sizing:</span>
              <span className="info-stat-value">
                {documentInfo?.chunk_size || 1000} chars ({documentInfo?.chunk_overlap || 200} overlap)
              </span>
            </div>
            <div className="info-stat-item">
              <span className="info-stat-label">Index Status:</span>
              <span className="status-badge-inline green">
                {documentInfo?.status || 'Indexed & Ready'}
              </span>
            </div>
          </div>
        </div>

        {/* Retrieval Pipeline Card */}
        <div className="info-card glass-card">
          <div className="info-card-header">
            <Database size={16} className="card-header-icon" />
            <span>Retrieval Pipeline</span>
          </div>
          <ul className="pipeline-steps-list">
            <li>
              <div className="step-num">1</div>
              <div className="step-content">
                <strong>Dense Vector Search</strong>
                <span>FAISS IndexFlatIP (384-dim normalized)</span>
              </div>
            </li>
            <li>
              <div className="step-num">2</div>
              <div className="step-content">
                <strong>Lexical Keyword Search</strong>
                <span>BM25 Okapi with custom stopwords</span>
              </div>
            </li>
            <li>
              <div className="step-num">3</div>
              <div className="step-content">
                <strong>Rank Fusion</strong>
                <span>Reciprocal Rank Fusion (RRF, k=60)</span>
              </div>
            </li>
            <li>
              <div className="step-num">4</div>
              <div className="step-content">
                <strong>Passage Reranking</strong>
                <span>Cross-Encoder ms-marco-MiniLM-L-6-v2</span>
              </div>
            </li>
          </ul>
        </div>

        {/* Model Architecture Card */}
        <div className="info-card glass-card">
          <div className="info-card-header">
            <Cpu size={16} className="card-header-icon" />
            <span>AI Models</span>
          </div>
          <div className="info-stat-list">
            <div className="info-stat-item">
              <span className="info-stat-label">Local LLM:</span>
              <span className="info-stat-value font-bold code-font">
                {documentInfo?.llm_model || 'qwen2.5:1.5b'}
              </span>
            </div>
            <div className="info-stat-item">
              <span className="info-stat-label">Embedding:</span>
              <span className="info-stat-value code-font">
                {documentInfo?.embedding_model || 'all-MiniLM-L6-v2'}
              </span>
            </div>
            <div className="info-stat-item">
              <span className="info-stat-label">Reranker:</span>
              <span className="info-stat-value code-font">
                {documentInfo?.cross_encoder_model || 'ms-marco-MiniLM-L-6-v2'}
              </span>
            </div>
            <div className="info-stat-item">
              <span className="info-stat-label">Inference Mode:</span>
              <span className="info-stat-value">100% Local (Deterministic temp=0)</span>
            </div>
          </div>
        </div>

        {/* Connectivity & Health Card */}
        <div className="info-card glass-card">
          <div className="info-card-header">
            <Server size={16} className="card-header-icon" />
            <span>Backend Services</span>
          </div>
          <div className="info-stat-list">
            <div className="info-stat-item">
              <span className="info-stat-label">FastAPI Server:</span>
              <span className="status-badge-inline green">
                {healthInfo?.status === 'ok' ? 'Connected (Port 8000)' : 'Connecting...'}
              </span>
            </div>
            <div className="info-stat-item">
              <span className="info-stat-label">Ollama Engine:</span>
              <span className={`status-badge-inline ${healthInfo?.ollama_status === 'connected' ? 'green' : 'amber'}`}>
                {healthInfo?.ollama_status === 'connected' ? 'Online (Qwen 1.5B)' : 'Disconnected'}
              </span>
            </div>
            <div className="info-stat-item">
              <span className="info-stat-label">Memory Footprint:</span>
              <span className="info-stat-value">Optimized for 8 GB RAM</span>
            </div>
          </div>
        </div>
      </div>
    </aside>
  );
};
