import React, { useState } from 'react';
import { OutputToken, TokenHeatmapEntry } from '../lib/api';

interface OutputHighlighterProps {
  tokens: OutputToken[];
  tokenHeatmap?: TokenHeatmapEntry[];
  onChunkHover?: (chunkId: string | null) => void;
}

// Map a max-attribution score [0, 1] to a green opacity for grounded tokens.
function groundedColor(maxScore: number): string {
  const alpha = Math.round(maxScore * 180); // 0–180 out of 255
  return `rgba(92, 224, 92, ${(alpha / 255).toFixed(2)})`;
}

export const OutputHighlighter: React.FC<OutputHighlighterProps> = ({
  tokens,
  tokenHeatmap,
  onChunkHover,
}) => {
  const [hoveredPosition, setHoveredPosition] = useState<number | null>(null);
  const [showHeatmap, setShowHeatmap] = useState(false);

  if (tokens.length === 0 && (!tokenHeatmap || tokenHeatmap.length === 0)) {
    return (
      <div className="text-[#8a8a8a] italic">No output tokens available</div>
    );
  }

  const hasHeatmap = tokenHeatmap && tokenHeatmap.length > 0;

  return (
    <div>
      {/* Mode toggle — only shown when heatmap data is available */}
      {hasHeatmap && (
        <div className="flex gap-1 mb-3">
          <button
            onClick={() => setShowHeatmap(false)}
            className={`px-2 py-0.5 text-xs rounded transition-colors ${
              !showHeatmap
                ? 'bg-[#2a2a2a] text-[#e8e8e8] border border-[#4a4a4a]'
                : 'text-[#6a6a6a] hover:text-[#8a8a8a]'
            }`}
          >
            Sentence view
          </button>
          <button
            onClick={() => setShowHeatmap(true)}
            className={`px-2 py-0.5 text-xs rounded transition-colors ${
              showHeatmap
                ? 'bg-[#1a2a1a] text-[#5ce05c] border border-[#3a6a3a]'
                : 'text-[#6a6a6a] hover:text-[#8a8a8a]'
            }`}
            title="Token-level attention heatmap from attention rollout (local HF model)"
          >
            Token heatmap
          </button>
        </div>
      )}

      {/* ── Token heatmap view ── */}
      {showHeatmap && hasHeatmap && (
        <div className="text-sm leading-relaxed text-[#e8e8e8] font-mono break-words">
          {tokenHeatmap!.map((entry) => {
            const maxScore = Math.max(0, ...Object.values(entry.chunk_attributions));
            const topChunkId = Object.entries(entry.chunk_attributions).reduce<[string, number] | null>(
              (prev, curr) => (!prev || curr[1] > prev[1] ? curr : prev),
              null
            )?.[0] ?? null;

            const isHovered = hoveredPosition === entry.position;
            const bg = isHovered ? 'rgba(92, 224, 92, 0.4)' : groundedColor(maxScore);

            return (
              <span
                key={entry.position}
                onMouseEnter={() => {
                  setHoveredPosition(entry.position);
                  onChunkHover?.(topChunkId);
                }}
                onMouseLeave={() => {
                  setHoveredPosition(null);
                  onChunkHover?.(null);
                }}
                style={{ backgroundColor: bg }}
                className="transition-colors rounded-sm"
                title={
                  `Token: "${entry.text}"\n` +
                  Object.entries(entry.chunk_attributions)
                    .filter(([, s]) => s > 0.01)
                    .sort(([, a], [, b]) => b - a)
                    .slice(0, 3)
                    .map(([id, s]) => `${id.substring(0, 8)}: ${(s * 100).toFixed(0)}%`)
                    .join('\n')
                }
              >
                {entry.text}
              </span>
            );
          })}
        </div>
      )}

      {/* ── Sentence view (original) ── */}
      {(!showHeatmap || !hasHeatmap) && (
        <div className="text-sm leading-relaxed text-[#e8e8e8]">
          {tokens.map((token) => (
            <span
              key={`${token.position}`}
              onMouseEnter={() => {
                setHoveredPosition(token.position);
                const topChunk = Object.entries(token.chunk_attributions).reduce<[string, number] | null>(
                  (prev, curr) =>
                    !prev || curr[1] > prev[1] ? [curr[0], curr[1]] : prev,
                  null
                );
                if (topChunk) {
                  onChunkHover?.(topChunk[0]);
                }
              }}
              onMouseLeave={() => {
                setHoveredPosition(null);
                onChunkHover?.(null);
              }}
              className={`transition-colors ${
                token.is_hallucinated
                  ? 'bg-[#e05c5c]/30 text-[#ff9999] font-medium'
                  : ''
              } ${
                hoveredPosition === token.position
                  ? 'bg-[#e05c5c]/50 px-1'
                  : ''
              }`}
              title={
                token.is_hallucinated
                  ? `Hallucination score: ${(token.hallucination_score * 100).toFixed(0)}%\nAttributions: ${Object.entries(token.chunk_attributions)
                      .filter(([, score]) => score > 0)
                      .map(
                        ([chunkId, score]) =>
                          `${chunkId.substring(0, 8)}: ${(score * 100).toFixed(0)}%`
                      )
                      .join(', ')}`
                  : undefined
              }
            >
              {token.text}
            </span>
          ))}
        </div>
      )}
    </div>
  );
};
