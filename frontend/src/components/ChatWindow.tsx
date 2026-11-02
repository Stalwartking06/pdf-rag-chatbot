import React, { useRef, useEffect } from 'react';
import { Menu, PanelRight, Sparkles, AlertTriangle, RefreshCw } from 'lucide-react';
import type { ChatMessage as ChatMessageType, DocumentInfo, HealthInfo } from '../types/chat';
import { ChatMessage } from './ChatMessage';
import { WelcomeScreen } from './WelcomeScreen';
import { ChatInput } from './ChatInput';

interface ChatWindowProps {
  messages: ChatMessageType[];
  isLoading: boolean;
  onSendMessage: (question: string) => void;
  onToggleSidebar: () => void;
  onToggleInfoPanel: () => void;
  isInfoPanelOpen: boolean;
  documentInfo: DocumentInfo | null;
  healthInfo: HealthInfo | null;
  error: string | null;
  onRetryConnection: () => void;
}

export const ChatWindow: React.FC<ChatWindowProps> = ({
  messages,
  isLoading,
  onSendMessage,
  onToggleSidebar,
  onToggleInfoPanel,
  isInfoPanelOpen,
  documentInfo,
  healthInfo,
  error,
  onRetryConnection,
}) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to the bottom whenever a new message is added or loading state changes
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  return (
    <main className="chat-window-container">
      {/* Top Navigation Bar */}
      <header className="chat-header glass-panel">
        <div className="header-left">
          <button
            type="button"
            className="header-icon-btn mobile-only"
            onClick={onToggleSidebar}
            aria-label="Open sidebar navigation"
            title="Toggle menu"
          >
            <Menu size={20} />
          </button>

          <div className="header-title-block">
            <div className="header-title-row">
              <span className="header-title">Policy Intelligence Assistant</span>
              <span className="live-status-pill">
                <span className={`pulse-dot ${healthInfo?.status === 'ok' ? 'green' : 'amber'}`} />
                <span>{healthInfo?.status === 'ok' ? 'Online' : 'Connecting'}</span>
              </span>
            </div>
            <div className="header-subtitle">
              Grounded in <strong>{documentInfo?.filename || 'company_policy.pdf'}</strong> ({documentInfo?.total_pages || 109} Pages)
            </div>
          </div>
        </div>

        <div className="header-right">
          <button
            type="button"
            className={`header-icon-btn ${isInfoPanelOpen ? 'active' : ''}`}
            onClick={onToggleInfoPanel}
            aria-label="Toggle system and document information panel"
            title="Document & Model Telemetry"
          >
            <PanelRight size={20} />
          </button>
        </div>
      </header>

      {/* Backend connection error banner if needed */}
      {error && (
        <div className="error-banner glass-card" role="alert">
          <div className="error-banner-content">
            <AlertTriangle size={18} className="error-banner-icon" />
            <span>{error}</span>
          </div>
          <button
            type="button"
            className="retry-btn"
            onClick={onRetryConnection}
            title="Retry connecting to backend"
          >
            <RefreshCw size={14} />
            <span>Retry</span>
          </button>
        </div>
      )}

      {/* Messages Scroll Area */}
      <div className="messages-scroll-area">
        {messages.length === 0 ? (
          <WelcomeScreen
            onSelectQuestion={onSendMessage}
            documentFilename={documentInfo?.filename}
          />
        ) : (
          <div className="messages-list">
            {messages.map((msg) => (
              <ChatMessage key={msg.id} message={msg} />
            ))}

            {/* Loading / Typing indicator */}
            {isLoading && (
              <div className="chat-message-row assistant-row loading-row">
                <div className="chat-avatar-container">
                  <div className="assistant-avatar glowing">
                    <Sparkles size={18} />
                  </div>
                </div>
                <div className="chat-bubble-container">
                  <div className="message-header-info">
                    <span className="author-name">PDF RAG Assistant</span>
                    <span className="retrieval-mode-badge mode-thinking">Searching & Synthesizing</span>
                  </div>
                  <div className="message-bubble assistant-bubble typing-bubble">
                    <div className="typing-indicator">
                      <span className="typing-dot" />
                      <span className="typing-dot" />
                      <span className="typing-dot" />
                    </div>
                    <span className="typing-status-text">
                      Retrieving excerpts & generating grounded answer...
                    </span>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Fixed Bottom Input Area */}
      <div className="bottom-input-container">
        <ChatInput
          onSendMessage={onSendMessage}
          isLoading={isLoading}
          disabled={Boolean(error && healthInfo?.status !== 'ok')}
        />
      </div>
    </main>
  );
};
