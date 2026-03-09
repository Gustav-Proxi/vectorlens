import React, { useState } from 'react';
import { OutputToken } from '../lib/api';

interface OutputHighlighterProps {
  tokens: OutputToken[];
  onChunkHover?: (chunkId: string | null) => void;
}

export const OutputHighlighter: React.FC<OutputHighlighterProps> = ({
  tokens,
  onChunkHover,
}) => {
  const [hoveredTokenPosition, setHoveredTokenPosition] = useState<
    number | null
  >(null);

  if (tokens.length === 0) {
    return (
      <div className="text-[#8a8a8a] italic">No output tokens available</div>
    );
  }

  return (
    <div className="text-sm leading-relaxed text-[#e8e8e8]">
      {tokens.map((token) => (
        <span
          key={`${token.position}`}
          onMouseEnter={() => {
            setHoveredTokenPosition(token.position);
            // Show which chunks contribute to this token
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
            setHoveredTokenPosition(null);
            onChunkHover?.(null);
          }}
          className={`transition-colors ${
            token.is_hallucinated
              ? 'bg-[#e05c5c]/30 text-[#ff9999] font-medium'
              : ''
          } ${
            hoveredTokenPosition === token.position
              ? 'bg-[#e05c5c]/50 px-1'
              : ''
          }`}
          title={
            token.is_hallucinated
              ? `Hallucination score: ${(token.hallucination_score * 100).toFixed(0)}%\nAttributions: ${Object.entries(token.chunk_attributions)
                  .filter(([_, score]) => score > 0)
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
  );
};
