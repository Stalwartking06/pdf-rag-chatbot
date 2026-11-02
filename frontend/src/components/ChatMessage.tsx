import React, { useState } from 'react';
import { User, Sparkles, AlertCircle, Layers, CheckCircle2, ChevronDown, ChevronUp } from 'lucide-react';
import type { ChatMessage as ChatMessageType } from '../types/chat';
import { SourceCard } from './SourceCard';

interface ChatMessageProps {
  message: ChatMessageType;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  const isUser = message.role === 'user';
  const [showTelemetry, setShowTelemetry] = useState<boolean>(false);

  // Format markdown-like bullet points and linebreaks cleanly
  const renderFormattedContent = (content: string) => {
    const lines = content.split('\n');
    return lines.map((line, idx) => {
      const trimmed = line.trim();
      if (!trimmed) {
        return <div key={idx} className="content-spacer" />;
      }
      if (trimmed.startsWith('- ') || trimmed.startsWith('• ') || trimmed.startsWith('* ')) {
        return (
          <li key={idx} className="content-bullet">
            {trimmed.substring(2)}
          </li>
        );
      }
      if (/^\d+\.\s/.test(trimmed)) {
        return (
          <div key={idx} className="content-numbered-line">
            {trimmed}
          </div>
        );
      }
      return (
        <p key={idx} className="content-paragraph">
          {trimmed}
        </p>
      );
    });
  };

  return (
    <div className={`chat-message-row ${isUser ? 'user-row' : 'assistant-row'}`}>
      <div className="chat-avatar-container">
        {isUser ? (
          <div className="user-avatar" title="You">
            <User size={18} />
          </div>
        ) : (
          <div className="assistant-avatar" title="PDF RAG Assistant">
            <Sparkles size={18} />
          </div>
        )}
      </div>

      <div className="chat-bubble-container">
        {/* Author header with timestamp and badges */}
        <div className="message-header-info">
          <span className="author-name">
            {isUser ? 'You' : 'PDF RAG Assistant'}
          </span>
          <span className="message-timestamp">{message.timestamp}</span>

          {!isUser && message.retrieval_mode && (
            <span className={`retrieval-mode-badge mode-${message.retrieval_mode}`}>
              {message.retrieval_mode === 'exact_lookup' && 'Exact Role Match'}
              {message.retrieval_mode === 'clarification' && 'Clarification'}
              {message.retrieval_mode === 'hybrid_rag' && 'Hybrid RAG + Reranked'}
            </span>
          )}
        </div>

        {/* Spelling Correction note if applicable */}
        {!isUser && message.corrected_question && (
          <div className="corrected-question-banner">
            <CheckCircle2 size={13} />
            <span>Interpreted query as: <strong>"{message.corrected_question}"</strong></span>
          </div>
        )}

        {/* Main message bubble content */}
        <div className={`message-bubble ${isUser ? 'user-bubble' : 'assistant-bubble'} ${message.isError ? 'error-bubble' : ''}`}>
          {message.isError && (
            <div className="error-icon-wrapper">
              <AlertCircle size={16} />
            </div>
          )}
          <div className="message-text">
            {renderFormattedContent(message.content)}
          </div>
        </div>

        {/* Retrieval Sources Citations Section */}
        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="message-sources-section">
            <div className="sources-section-header">
              <Layers size={14} />
              <span>Referenced Policy Sources ({message.sources.length})</span>
            </div>
            <div className="sources-cards-list">
              {message.sources.map((source, index) => (
                <SourceCard
                  key={`${source.chunk_id}-${index}`}
                  source={source}
                  index={index + 1}
                />
              ))}
            </div>
          </div>
        )}

        {/* Retrieval Telemetry Details Toggle */}
        {!isUser && message.debug_info && (
          <div className="telemetry-section">
            <button
              type="button"
              className="telemetry-toggle-btn"
              onClick={() => setShowTelemetry(!showTelemetry)}
              title="Inspect low-level retrieval candidates and Cross-Encoder scoring"
            >
              <span>{showTelemetry ? 'Hide Retrieval Telemetry' : 'View Retrieval Telemetry'}</span>
              {showTelemetry ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            </button>

            {showTelemetry && (
              <div className="telemetry-content glass-subcard">
                <div className="telemetry-item">
                  <span className="telemetry-label">Expanded Query:</span>
                  <span className="telemetry-val">"{message.debug_info.expanded_query}"</span>
                </div>
                <div className="telemetry-grid">
                  <div className="stat-pill">
                    <strong>FAISS Candidates:</strong> {message.debug_info.faiss_candidates_count}
                  </div>
                  <div className="stat-pill">
                    <strong>BM25 Candidates:</strong> {message.debug_info.bm25_candidates_count}
                  </div>
                  <div className="stat-pill">
                    <strong>RRF Merged:</strong> {message.debug_info.hybrid_candidates_count}
                  </div>
                </div>

                {message.debug_info.cross_encoder_scores && message.debug_info.cross_encoder_scores.length > 0 && (
                  <div className="telemetry-scores-list">
                    <span className="telemetry-label">Top Cross-Encoder Logits:</span>
                    <ul>
                      {message.debug_info.cross_encoder_scores.map((s, idx) => (
                        <li key={idx}>
                          Chunk #{s.chunk_id} (Page {s.page_number}): <strong>{s.score.toFixed(4)}</strong>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
