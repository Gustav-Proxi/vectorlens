import React from 'react';
import { RetrievedChunk } from '../lib/api';

interface ChunkCardProps {
  chunk: RetrievedChunk;
  isHighlighted: boolean;
  rank: number;
  onHover?: (chunkId: string | null) => void;
}

function getAttributionColor(score: number, causedHallucination: boolean): string {
  if (causedHallucination) return '#e05c5c';
  if (score > 0.6) return '#5ce05c';
  if (score > 0.3) return '#d4a574';
  return '#3a3a3a';
}

export const ChunkCard: React.FC<ChunkCardProps> = ({ chunk, isHighlighted, rank, onHover }) => {
  const pct = Math.round(chunk.attribution_score * 100);
  const barColor = getAttributionColor(chunk.attribution_score, chunk.caused_hallucination);
  const docTitle = (chunk.metadata?.document_title as string) ?? '';

  return (
    <div
      onMouseEnter={() => onHover?.(chunk.chunk_id)}
      onMouseLeave={() => onHover?.(null)}
      className={`flex gap-3 p-3 rounded border transition-all cursor-default ${
        isHighlighted
          ? 'border-[#e05c5c] bg-[#2a1a1a] shadow-md'
          : 'border-[#2a2a2a] bg-[#1a1a1a] hover:border-[#3a3a3a]'
      }`}
    >
      {/* Rank + attribution bar */}
      <div className="flex flex-col items-center gap-1 w-8 flex-shrink-0">
        <span className="text-[#6a6a6a] text-xs font-mono">#{rank}</span>
        <div className="flex-1 w-2 bg-[#0f0f0f] rounded-full overflow-hidden min-h-8">
          <div
            className="w-full rounded-full transition-all"
            style={{ height: `${Math.max(pct, 4)}%`, backgroundColor: barColor }}
          />
        </div>
        <span className="text-xs font-bold" style={{ color: barColor }}>{pct}%</span>
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        {/* Header row */}
        <div className="flex items-center gap-2 mb-1 flex-wrap">
          {chunk.caused_hallucination && (
            <span className="px-1.5 py-0.5 bg-[#3a1a1a] border border-[#e05c5c] text-[#e8a8a8] text-xs rounded">
              hallucinated
            </span>
          )}
          <span className="px-1.5 py-0.5 bg-[#0f0f0f] text-[#6a6a6a] text-xs rounded font-mono">
            sim {chunk.score.toFixed(3)}
          </span>
          {docTitle && (
            <span className="text-[#6a6a6a] text-xs truncate max-w-40" title={docTitle}>
              {docTitle}
            </span>
          )}
        </div>

        {/* Chunk text */}
        <p className="text-xs text-[#b8b8b8] leading-relaxed line-clamp-3">
          {chunk.text}
        </p>
      </div>
    </div>
  );
};
