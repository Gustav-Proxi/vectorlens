import React, { useState } from 'react';
import { Session } from '../lib/api';
import { OutputHighlighter } from './OutputHighlighter';
import { ChunkCard } from './ChunkCard';

interface AttributionViewProps {
  session: Session | null;
  loading: boolean;
  error: Error | null;
  isCached?: boolean;
}

export const AttributionView: React.FC<AttributionViewProps> = ({
  session,
  loading,
  error,
  isCached = false,
}) => {
  const [hoveredChunkId, setHoveredChunkId] = useState<string | null>(null);

  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="bg-[#2a1a1a] border border-[#e05c5c] rounded p-6 text-center">
          <p className="text-[#e8a8a8] font-medium mb-2">Error loading session</p>
          <p className="text-[#8a8a8a] text-sm">{error.message}</p>
        </div>
      </div>
    );
  }

  if (loading || !session) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-[#e05c5c] mb-4" />
          <p className="text-[#8a8a8a]">Loading session...</p>
        </div>
      </div>
    );
  }

  // Pick the attribution for the most meaningful response:
  // prefer longest output that isn't JSON (JSON = internal agent call, not the answer)
  const isJson = (text: string) => text.trimStart().startsWith('{') || text.trimStart().startsWith('```json');
  const allAttributions = session.attributions ?? [];
  const attribution =
    allAttributions
      .filter(a => a.output_tokens?.length > 0)
      .sort((a, b) => {
        const textA = a.output_tokens.map((t: {text: string}) => t.text).join(' ');
        const textB = b.output_tokens.map((t: {text: string}) => t.text).join(' ');
        const aIsJson = isJson(textA);
        const bIsJson = isJson(textB);
        if (aIsJson !== bIsJson) return aIsJson ? 1 : -1; // non-JSON first
        return textB.length - textA.length; // then longest
      })[0] ?? allAttributions[0];

  const llmResponse = session.llm_responses?.[0];

  if (!attribution && llmResponse) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-[#e05c5c] mb-4" />
          <p className="text-[#8a8a8a]">Analyzing...</p>
          <p className="text-[#6a6a6a] text-sm mt-2">Computing attributions</p>
        </div>
      </div>
    );
  }

  if (!attribution) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <p className="text-[#8a8a8a] mb-2">Waiting for RAG pipeline to run...</p>
          <p className="text-[#6a6a6a] text-sm">
            Queries: {session.vector_queries?.length ?? 0} | Responses:{' '}
            {session.llm_responses?.length ?? 0}
          </p>
        </div>
      </div>
    );
  }

  const sortedChunks = [...attribution.chunks].sort(
    (a, b) => b.attribution_score - a.attribution_score
  );

  const groundednessPercent = Math.round(
    attribution.overall_groundedness * 100
  );
  const groundednessColor =
    attribution.overall_groundedness > 0.7
      ? '#5ce05c'
      : attribution.overall_groundedness > 0.4
        ? '#8a8a8a'
        : '#e05c5c';

  return (
    <div className="flex flex-col h-full bg-[#0f0f0f]">
      {/* Header */}
      <div className="border-b border-[#2a2a2a] p-6 bg-[#1a1a1a]">
        {isCached && (
          <div className="mb-4 text-xs text-[#8a8a8a] px-3 py-2 bg-[#0f0f0f] rounded border border-[#2a2a2a]">
            📋 Cached session — viewing from local storage
          </div>
        )}
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-sm font-mono text-[#b8b8b8] mb-1">
              {session.id}
            </h2>
            <p className="text-xs text-[#8a8a8a]">
              {session.vector_queries?.length ?? 0} vector queries • {session.llm_responses?.length ?? 0} responses •{' '}
              {attribution.chunks.length} retrieved chunks
            </p>
          </div>
        </div>

        {/* Groundedness Score */}
        <div>
          <div className="flex justify-between items-center mb-2">
            <span className="text-xs font-medium text-[#8a8a8a]">
              Overall Groundedness
            </span>
            <span
              className="text-sm font-bold"
              style={{ color: groundednessColor }}
            >
              {groundednessPercent}%
            </span>
          </div>
          <div className="h-2 bg-[#0f0f0f] rounded overflow-hidden">
            <div
              className="h-full transition-all"
              style={{
                width: `${attribution.overall_groundedness * 100}%`,
                backgroundColor: groundednessColor,
              }}
            />
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-hidden flex flex-col">
        {/* Output Panel */}
        <div className="flex-1 overflow-y-auto p-6 border-b border-[#2a2a2a]">
          <h3 className="text-xs font-semibold text-[#8a8a8a] uppercase tracking-wide mb-4">
            LLM Output
          </h3>
          <div className="bg-[#1a1a1a] rounded p-4 border border-[#2a2a2a]">
            <OutputHighlighter
              tokens={attribution.output_tokens}
              onChunkHover={setHoveredChunkId}
            />
          </div>
        </div>

        {/* Chunks Panel */}
        <div className="flex-1 overflow-hidden flex flex-col">
          <div className="p-6 pb-4 border-b border-[#2a2a2a]">
            <h3 className="text-xs font-semibold text-[#8a8a8a] uppercase tracking-wide">
              Retrieved Chunks ({sortedChunks.length})
            </h3>
          </div>
          <div className="flex-1 overflow-x-auto overflow-y-hidden">
            <div className="flex gap-4 p-6 pb-6">
              {sortedChunks.map((chunk) => (
                <ChunkCard
                  key={chunk.chunk_id}
                  chunk={chunk}
                  isHighlighted={hoveredChunkId === chunk.chunk_id}
                  onHover={setHoveredChunkId}
                />
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
