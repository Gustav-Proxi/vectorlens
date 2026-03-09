import React, { useState } from 'react';
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
  return '#555555';
}

export const ChunkCard: React.FC<ChunkCardProps> = ({ chunk, isHighlighted, rank, onHover }) => {
  const [expanded, setExpanded] = useState(false);
  const pct = Math.round(chunk.attribution_score * 100);
  const barColor = getAttributionColor(chunk.attribution_score, chunk.caused_hallucination);
  const docTitle = (chunk.metadata?.document_title as string) ?? '';
  const chunkType = (chunk.metadata?.chunk_type as string) ?? '';
  const sectionPath = (chunk.metadata?.section_path as string) ?? '';
  const vectorScore = chunk.metadata?.vector_score as number | undefined;
  const keywordScore = chunk.metadata?.keyword_score as number | undefined;

  return (
    <div
      onMouseEnter={() => onHover?.(chunk.chunk_id)}
      onMouseLeave={() => onHover?.(null)}
      className={`rounded border transition-all ${
        isHighlighted
          ? 'border-[#e05c5c] bg-[#1e1212] shadow-lg shadow-[#e05c5c]/10'
          : 'border-[#2a2a2a] bg-[#1a1a1a] hover:border-[#3a3a3a]'
      }`}
    >
      {/* Collapsed row — always visible */}
      <div
        className="flex gap-3 p-3 cursor-pointer select-none"
        onClick={() => setExpanded(e => !e)}
      >
        {/* Rank + score */}
        <div className="flex flex-col items-center gap-0.5 w-8 flex-shrink-0">
          <span className="text-[#6a6a6a] text-xs font-mono">#{rank}</span>
          <div className="flex-1 w-1.5 bg-[#0f0f0f] rounded-full overflow-hidden min-h-6">
            <div
              className="w-full rounded-full transition-all"
              style={{ height: `${Math.max(pct, 6)}%`, backgroundColor: barColor }}
            />
          </div>
          <span className="text-xs font-bold tabular-nums" style={{ color: barColor }}>{pct}%</span>
        </div>

        {/* Content preview */}
        <div className="flex-1 min-w-0">
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
              <span className="text-[#6a6a6a] text-xs truncate max-w-48" title={docTitle}>
                {docTitle}
              </span>
            )}
          </div>
          <p className={`text-xs text-[#b8b8b8] leading-relaxed ${expanded ? '' : 'line-clamp-2'}`}>
            {chunk.text}
          </p>
        </div>

        {/* Expand toggle */}
        <div className="flex-shrink-0 self-start pt-1">
          <span className="text-[#4a4a4a] text-xs select-none">{expanded ? '▲' : '▼'}</span>
        </div>
      </div>

      {/* Expanded detail panel */}
      {expanded && (
        <div className="border-t border-[#2a2a2a] px-4 py-3 space-y-3">

          {/* Full text */}
          <div>
            <div className="text-xs font-semibold text-[#6a6a6a] uppercase tracking-wider mb-1">
              Full text
            </div>
            <p className="text-xs text-[#c8c8c8] leading-relaxed whitespace-pre-wrap bg-[#0f0f0f] rounded p-3 border border-[#2a2a2a]">
              {chunk.text}
            </p>
          </div>

          {/* Scores grid */}
          <div>
            <div className="text-xs font-semibold text-[#6a6a6a] uppercase tracking-wider mb-2">
              Scores
            </div>
            <div className="grid grid-cols-2 gap-2">
              <ScoreRow label="Attribution" value={`${pct}%`} color={barColor} />
              <ScoreRow label="Similarity" value={chunk.score.toFixed(4)} />
              {vectorScore != null && <ScoreRow label="Vector score" value={vectorScore.toFixed(4)} />}
              {keywordScore != null && <ScoreRow label="Keyword score" value={keywordScore.toFixed(4)} />}
            </div>
          </div>

          {/* Metadata */}
          <div>
            <div className="text-xs font-semibold text-[#6a6a6a] uppercase tracking-wider mb-2">
              Metadata
            </div>
            <div className="space-y-1">
              {docTitle && <MetaRow label="Document" value={docTitle} />}
              {chunkType && <MetaRow label="Type" value={chunkType} />}
              {sectionPath && <MetaRow label="Section" value={sectionPath} />}
              <MetaRow label="Chunk ID" value={chunk.chunk_id} mono />
              {Object.entries(chunk.metadata ?? {})
                .filter(([k]) => !['document_title','chunk_type','section_path','vector_score','keyword_score','hyde_score'].includes(k))
                .map(([k, v]) => (
                  <MetaRow key={k} label={k} value={String(v)} />
                ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

const ScoreRow = ({ label, value, color }: { label: string; value: string; color?: string }) => (
  <div className="flex justify-between items-center bg-[#0f0f0f] rounded px-2 py-1.5">
    <span className="text-xs text-[#6a6a6a]">{label}</span>
    <span className="text-xs font-mono font-bold" style={{ color: color ?? '#b8b8b8' }}>{value}</span>
  </div>
);

const MetaRow = ({ label, value, mono }: { label: string; value: string; mono?: boolean }) => (
  <div className="flex gap-2 text-xs">
    <span className="text-[#6a6a6a] flex-shrink-0 w-20">{label}</span>
    <span className={`text-[#b8b8b8] truncate ${mono ? 'font-mono text-[10px]' : ''}`} title={value}>{value}</span>
  </div>
);
