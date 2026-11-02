import React from 'react';
import { 
  Plus, 
  FileText, 
  Trash2, 
  Layers, 
  Cpu, 
  Database, 
  CheckCircle,
  X,
  MessageSquare
} from 'lucide-react';
import type { DocumentInfo, HealthInfo } from '../types/chat';

interface SidebarProps {
  documentInfo: DocumentInfo | null;
  healthInfo: HealthInfo | null;
  onNewChat: () => void;
  onClearChat: () => void;
  isOpen: boolean;
  onCloseMobile: () => void;
  hasMessages: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({
  documentInfo,
  healthInfo,
  onNewChat,
  onClearChat,
  isOpen,
  onCloseMobile,
  hasMessages,
}) => {
  return (
    <>
      {/* Mobile backdrop */}
      {isOpen && (
        <div 
          className="sidebar-backdrop mobile-only" 
          onClick={onCloseMobile}
          aria-hidden="true"
        />
      )}

      <aside className={`app-sidebar glass-panel ${isOpen ? 'open' : ''}`}>
        {/* Brand header */}
        <div className="sidebar-brand-header">
          <div className="brand-logo-group">
            <div className="brand-logo-icon">
              <Layers size={20} className="brand-icon-svg" />
            </div>
            <div className="brand-text">
              <span className="brand-title">PDF RAG Assistant</span>
              <span className="brand-version">v1.0 • Local AI</span>
            </div>
          </div>
          <button 
            type="button" 
            className="mobile-close-btn mobile-only"
            onClick={onCloseMobile}
            aria-label="Close sidebar"
          >
            <X size={18} />
          </button>
        </div>

        {/* Action: New Chat */}
        <div className="sidebar-action-section">
          <button
            type="button"
            className="new-chat-button"
            onClick={onNewChat}
          >
            <Plus size={16} />
            <span>New Chat Session</span>
          </button>
        </div>

        {/* Document Status Section */}
        <div className="sidebar-section">
          <span className="section-label">Target Document</span>
          <div className="document-info-card glass-card">
            <div className="doc-card-icon">
              <FileText size={20} />
            </div>
            <div className="doc-card-details">
              <span className="doc-filename">
                {documentInfo?.filename || 'company_policy.pdf'}
              </span>
              <div className="doc-status-badge">
                <span className="pulse-dot green" />
                <span>{documentInfo?.status || 'Indexed & Ready'}</span>
              </div>
            </div>
          </div>

          <div className="doc-meta-stats">
            <div className="stat-row">
              <span className="stat-name">Total Pages:</span>
              <span className="stat-val">{documentInfo?.total_pages || 109} Pages</span>
            </div>
            <div className="stat-row">
              <span className="stat-name">Total Chunks:</span>
              <span className="stat-val">{documentInfo?.total_chunks || 285} Chunks</span>
            </div>
          </div>
        </div>

        {/* Chat Session Status */}
        <div className="sidebar-section flex-grow">
          <span className="section-label">Active Session</span>
          <div className="session-card active glass-card">
            <MessageSquare size={16} />
            <div className="session-details">
              <span className="session-title">Current Conversation</span>
              <span className="session-sub">
                {hasMessages ? 'In progress' : 'Empty conversation'}
              </span>
            </div>
          </div>

          {hasMessages && (
            <button
              type="button"
              className="clear-chat-button"
              onClick={onClearChat}
              title="Reset conversation messages"
            >
              <Trash2 size={14} />
              <span>Clear Conversation</span>
            </button>
          )}
        </div>

        {/* Technology Architecture Section */}
        <div className="sidebar-section tech-stack-section">
          <span className="section-label">Engine Architecture</span>
          <div className="tech-badges-grid">
            <div className="tech-badge" title="Dense vector similarity search">
              <Database size={12} />
              <span>FAISS (all-MiniLM)</span>
            </div>
            <div className="tech-badge" title="Lexical keyword retrieval">
              <Layers size={12} />
              <span>BM25 Okapi</span>
            </div>
            <div className="tech-badge" title="Reciprocal Rank Fusion k=60">
              <CheckCircle size={12} />
              <span>RRF Fusion</span>
            </div>
            <div className="tech-badge" title="ms-marco-MiniLM cross-encoder reranker">
              <Cpu size={12} />
              <span>Cross-Encoder</span>
            </div>
            <div className="tech-badge tech-llm" title="Local Ollama Qwen 2.5:1.5b model">
              <Cpu size={12} />
              <span>Ollama {documentInfo?.llm_model || 'qwen2.5:1.5b'}</span>
            </div>
          </div>

          <div className="connection-status-pill">
            <span className={`status-indicator-dot ${healthInfo?.ollama_status === 'connected' ? 'online' : 'warn'}`} />
            <span>Ollama: {healthInfo?.ollama_status || 'Connecting...'}</span>
          </div>
        </div>
      </aside>
    </>
  );
};
