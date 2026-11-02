import React, { useState, useEffect, useCallback } from 'react';
import type { ChatMessage, DocumentInfo, HealthInfo } from './types/chat';
import { apiService } from './services/api';
import { Sidebar } from './components/Sidebar';
import { ChatWindow } from './components/ChatWindow';
import { InfoPanel } from './components/InfoPanel';

export const App: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [documentInfo, setDocumentInfo] = useState<DocumentInfo | null>(null);
  const [healthInfo, setHealthInfo] = useState<HealthInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState<boolean>(false);
  const [isInfoPanelOpen, setIsInfoPanelOpen] = useState<boolean>(() => {
    // Open info panel by default on desktop (> 1280px)
    return typeof window !== 'undefined' ? window.innerWidth >= 1280 : true;
  });

  // Fetch backend health and document metadata
  const loadSystemInfo = useCallback(async () => {
    try {
      setError(null);
      const [health, doc] = await Promise.all([
        apiService.getHealth(),
        apiService.getDocumentInfo()
      ]);
      setHealthInfo(health);
      setDocumentInfo(doc);
    } catch (err: any) {
      console.error('Failed to load system information:', err);
      const errorMsg = err?.response?.data?.detail || err?.message || 'Cannot connect to FastAPI backend server.';
      setError(`Backend Connection Notice: ${errorMsg}`);
    }
  }, []);

  useEffect(() => {
    loadSystemInfo();
    // Poll health periodically every 30 seconds
    const interval = setInterval(loadSystemInfo, 30000);
    return () => clearInterval(interval);
  }, [loadSystemInfo]);

  // Format current time helper (e.g. "12:15 PM")
  const getFormattedTime = () => {
    return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  // Send question handler
  const handleSendMessage = async (question: string) => {
    if (!question.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: question.trim(),
      timestamp: getFormattedTime(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);
    setError(null);

    try {
      const response = await apiService.sendChatQuestion(question.trim());

      const assistantMessage: ChatMessage = {
        id: `assistant-${Date.now()}`,
        role: 'assistant',
        content: response.answer,
        timestamp: getFormattedTime(),
        corrected_question: response.corrected_question,
        retrieval_mode: response.retrieval_mode,
        sources: response.sources,
        debug_info: response.debug_info,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err: any) {
      console.error('Error generating answer:', err);
      const detail = err?.response?.data?.detail || err?.message || 'Failed to communicate with RAG engine.';

      const errorMessage: ChatMessage = {
        id: `err-${Date.now()}`,
        role: 'assistant',
        content: `Error generating response: ${detail}. Please ensure the FastAPI server and Ollama are active.`,
        timestamp: getFormattedTime(),
        isError: true,
      };

      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // New Chat / Clear Chat Handler
  const handleResetChat = async () => {
    try {
      await apiService.resetChat();
    } catch (err) {
      console.warn('Backend reset acknowledged with error (ignored):', err);
    }
    setMessages([]);
    setIsSidebarOpen(false);
  };

  return (
    <div className="app-root-layout">
      {/* Left Sidebar */}
      <Sidebar
        documentInfo={documentInfo}
        healthInfo={healthInfo}
        onNewChat={handleResetChat}
        onClearChat={handleResetChat}
        isOpen={isSidebarOpen}
        onCloseMobile={() => setIsSidebarOpen(false)}
        hasMessages={messages.length > 0}
      />

      {/* Center Chat Area */}
      <ChatWindow
        messages={messages}
        isLoading={isLoading}
        onSendMessage={handleSendMessage}
        onToggleSidebar={() => setIsSidebarOpen((prev) => !prev)}
        onToggleInfoPanel={() => setIsInfoPanelOpen((prev) => !prev)}
        isInfoPanelOpen={isInfoPanelOpen}
        documentInfo={documentInfo}
        healthInfo={healthInfo}
        error={error}
        onRetryConnection={loadSystemInfo}
      />

      {/* Right Information Panel (Collapsible) */}
      <InfoPanel
        documentInfo={documentInfo}
        healthInfo={healthInfo}
        isOpen={isInfoPanelOpen}
        onClose={() => setIsInfoPanelOpen(false)}
      />
    </div>
  );
};

export default App;
