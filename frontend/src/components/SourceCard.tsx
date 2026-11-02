import React, { useState } from 'react';
import { ChevronDown, ChevronUp, FileText, Bookmark, Target } from 'lucide-react';
import type { SourceItem } from '../types/chat';

interface SourceCardProps {
  source: SourceItem;
  index: number;
}

export const SourceCard: React.FC<SourceCardProps> = ({ source, index }) => {
  const [isExpanded, setIsExpanded] = useState<boolean>(false);

  // Score formatting
  const formattedScore = typeof source.score === 'number' 
    ? (source.score > 0 ? `+${source.score.toFixed(3)}` : source.score.toFixed(3))
    : 'N/A';

  return (
    <div className={`source-card ${isExpanded ? 'expanded' : ''}`}>
      <button 
        type="button" 
        className="source-card-header"
        onClick={() => setIsExpanded(!isExpanded)}
        aria-expanded={isExpanded}
        title="Click to toggle source excerpt"
      >
        <div className="source-meta-left">
          <span className="source-index-badge">
            <Bookmark size={12} />
            Source [{index}]
          </span>
          <span className="source-page-badge">
            <FileText size={12} />
            Page {source.page_number}
          </span>
          <span className="source-chunk-badge">
            Chunk #{source.chunk_id}
          </span>
        </div>

        <div className="source-meta-right">
          <span className="source-score-badge" title="Cross-Encoder relevance logit score">
            <Target size={12} />
            Score: {formattedScore}
          </span>
          <span className="source-expand-icon">
            {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </span>
        </div>
      </button>

      {isExpanded && (
        <div className="source-card-body">
          <div className="source-filename">
            <strong>Document:</strong> {source.source}
          </div>
          <div className="source-excerpt">
            <p>{source.text}</p>
          </div>
        </div>
      )}
    </div>
  );
};
