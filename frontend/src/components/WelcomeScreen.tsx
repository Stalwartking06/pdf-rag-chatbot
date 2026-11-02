import React from 'react';
import { Compass, UserCheck, ShieldCheck, Calendar, Briefcase, FileText } from 'lucide-react';

interface WelcomeScreenProps {
  onSelectQuestion: (question: string) => void;
  documentFilename?: string;
}

export const WelcomeScreen: React.FC<WelcomeScreenProps> = ({ 
  onSelectQuestion, 
  documentFilename = 'company_policy.pdf' 
}) => {
  const suggestedQuestions = [
    {
      title: 'Company Mission',
      question: 'What is the company mission?',
      icon: <Compass size={18} className="suggested-icon icon-cyan" />,
      desc: 'Learn about organizational goals and nurse triage leadership'
    },
    {
      title: 'Executive Leadership (CEO)',
      question: 'Who is the CEO?',
      icon: <UserCheck size={18} className="suggested-icon icon-indigo" />,
      desc: 'Chairperson and Chief Executive Officer details'
    },
    {
      title: 'Medical Director (COO)',
      question: 'Who is the COO?',
      icon: <ShieldCheck size={18} className="suggested-icon icon-violet" />,
      desc: 'Chief Operating Officer and Medical Direction'
    },
    {
      title: 'Leave & Attendance Policy',
      question: 'What is the leave policy?',
      icon: <Calendar size={18} className="suggested-icon icon-amber" />,
      desc: 'Calling in sick, leave procedures, and health safety'
    },
    {
      title: 'Employee Responsibilities',
      question: 'What are employee responsibilities?',
      icon: <Briefcase size={18} className="suggested-icon icon-emerald" />,
      desc: 'Workplace conduct, job performance, and company standards'
    }
  ];

  return (
    <div className="welcome-screen-container">
      <div className="welcome-hero-card glass-panel">
        <div className="hero-icon-badge">
          <FileText size={32} className="hero-icon-svg" />
        </div>
        <h1 className="welcome-title">PDF RAG Assistant</h1>
        <p className="welcome-subtitle">
          Intelligent conversational assistant for{' '}
          <span className="doc-highlight">{documentFilename}</span>.
        </p>
        <p className="welcome-description">
          Ask questions about policies, workplace procedures, or leadership. Every response is strictly grounded in retrieved document excerpts using hybrid vector search (FAISS) + keyword matching (BM25) and local LLM synthesis.
        </p>

        <div className="suggested-questions-container">
          <div className="suggested-header">
            <span>Suggested Inquiries</span>
            <span className="suggested-subtext">Click any question to start</span>
          </div>

          <div className="suggested-grid">
            {suggestedQuestions.map((item, idx) => (
              <button
                key={idx}
                type="button"
                className="suggested-card glass-card"
                onClick={() => onSelectQuestion(item.question)}
              >
                <div className="suggested-card-top">
                  {item.icon}
                  <span className="suggested-card-title">{item.title}</span>
                </div>
                <div className="suggested-card-query">"{item.question}"</div>
                <div className="suggested-card-desc">{item.desc}</div>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
