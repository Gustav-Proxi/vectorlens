import React from 'react';
import { RetrievedChunk } from '../lib/api';

interface ChunkCardProps {
  chunk: RetrievedChunk;
  isHighlighted: boolean;
  onHover?: (chunkId: string | null) => void;
}

function getAttributionColor(
  attributionScore: number,
  causedHallucination: boolean
): string {
  if (causedHallucination) {
    return 'bg-[#e05c5c]';
  }
  if (attributionScore > 0.7) {
    return 'bg-[#5ce05c]';
  }
  if (attributionScore > 0.4) {
    return 'bg-[#8a8a8a]';
  }
  return 'bg-[#3a3a3a]';
}

export const ChunkCard: React.FC<ChunkCardProps> = ({
  chunk,
  isHighlighted,
  onHover,
}) => {
  const displayText =
    chunk.text.length > 150 ? chunk.text.substring(0, 150) + '...' : chunk.text;

  return (
    <div
      onMouseEnter={() => onHover?.(chunk.chunk_id)}
      onMouseLeave={() => onHover?.(null)}
      className={`bg-[#1a1a1a] rounded p-3 min-w-96 flex-shrink-0 border transition-all cursor-default ${
        isHighlighted
          ? 'border-[#e05c5c] shadow-lg shadow-[#e05c5c]/20'
          : 'border-[#2a2a2a]'
      }`}
    >
      {/* Attribution Score Bar */}
      <div className="mb-2">
        <div className="flex justify-between items-center mb-1">
          <span className="text-xs font-medium text-[#8a8a8a]">Attribution</span>
          <span className="text-xs text-[#b8b8b8]">
            {(chunk.attribution_score * 100).toFixed(0)}%
          </span>
        </div>
        <div className="h-2 bg-[#0f0f0f] rounded overflow-hidden">
          <div
            className={`h-full transition-all ${getAttributionColor(
              chunk.attribution_score,
              chunk.caused_hallucination
            )}`}
            style={{ width: `${chunk.attribution_score * 100}%` }}
          />
        </div>
      </div>

      {/* Badges */}
      <div className="flex gap-2 mb-2 flex-wrap">
        {chunk.caused_hallucination && (
          <span className="inline-block px-2 py-1 bg-[#3a1a1a] border border-[#e05c5c] text-[#e8a8a8] text-xs rounded font-medium">
            Hallucination
          </span>
        )}
        <span className="inline-block px-2 py-1 bg-[#1a2a2a] border border-[#5ce05c] text-[#a8e8a8] text-xs rounded">
          Score: {chunk.score.toFixed(3)}
        </span>
      </div>

      {/* Text Preview */}
      <p className="text-xs text-[#b8b8b8] leading-relaxed break-words">
        "{displayText}"
      </p>

      {/* Metadata */}
      {Object.keys(chunk.metadata).length > 0 && (
        <div className="mt-2 pt-2 border-t border-[#2a2a2a]">
          <div className="text-xs text-[#8a8a8a]">
            {Object.entries(chunk.metadata).map(([key, value]) => (
              <div key={key}>
                <span className="text-[#6a6a6a]">{key}:</span>{' '}
                <span className="text-[#b8b8b8]">
                  {String(value).substring(0, 30)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
