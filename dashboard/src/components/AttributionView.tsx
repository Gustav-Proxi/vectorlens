import React, { useState } from 'react';
import { Session } from '../lib/api';
import { OutputHighlighter } from './OutputHighlighter';
import { ChunkCard } from './ChunkCard';
import { EmbeddingScatter } from './EmbeddingScatter';

const API_BASE = '/api';

async function runDeepAnalysis(sessionId: string, requestId?: string): Promise<void> {
  const url = requestId
    ? `${API_BASE}/sessions/${sessionId}/analyze?request_id=${requestId}`
    : `${API_BASE}/sessions/${sessionId}/analyze`;
  const resp = await fetch(url, { method: 'POST' });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail ?? resp.statusText);
  }
}

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
  const [analyzing, setAnalyzing] = useState(false);
  const [analyzeError, setAnalyzeError] = useState<string | null>(null);
  const [analyzeSuccess, setAnalyzeSuccess] = useState(false);
  const [rightTab, setRightTab] = useState<'chunks' | 'scatter'>('chunks');

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

  // Pick the best attribution: longest non-JSON response
  const isJson = (text: string) => text.trimStart().startsWith('{') || text.trimStart().startsWith('```json');
  const allAttributions = session.attributions ?? [];
  const attribution =
    allAttributions
      .filter(a => (a.output_tokens?.length ?? 0) > 0)
      .sort((a, b) => {
        const textA = a.output_tokens.map((t: { text: string }) => t.text).join(' ');
        const textB = b.output_tokens.map((t: { text: string }) => t.text).join(' ');
        const aIsJson = isJson(textA);
        const bIsJson = isJson(textB);
        if (aIsJson !== bIsJson) return aIsJson ? 1 : -1;
        return textB.length - textA.length;
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
            {session.vector_queries?.length ?? 0} queries · {session.llm_responses?.length ?? 0} responses
          </p>
        </div>
      </div>
    );
  }

  const sortedChunks = [...attribution.chunks].sort(
    (a, b) => b.attribution_score - a.attribution_score
  );

  const groundednessPercent = Math.round(attribution.overall_groundedness * 100);
  const groundednessColor =
    attribution.overall_groundedness > 0.7 ? '#5ce05c'
    : attribution.overall_groundedness > 0.4 ? '#d4a574'
    : '#e05c5c';

  const hallucinated = (attribution.output_tokens ?? []).filter(t => t.is_hallucinated).length;
  const total = (attribution.output_tokens ?? []).length;

  return (
    <div className="flex flex-col h-full bg-[#0f0f0f] overflow-hidden">

      {/* ── Top bar ── */}
      <div className="flex-shrink-0 border-b border-[#2a2a2a] bg-[#1a1a1a] px-6 py-4">
        <div className="flex items-center justify-between gap-6">
          {/* Session meta */}
          <div className="min-w-0">
            {isCached && (
              <span className="text-xs text-[#6a6a6a] mr-3">📋 cached</span>
            )}
            <span className="text-xs font-mono text-[#6a6a6a]">{session.id.substring(0, 12)}…</span>
            <span className="text-xs text-[#6a6a6a] ml-3">
              {session.vector_queries?.length ?? 0} queries · {session.llm_responses?.length ?? 0} responses · {sortedChunks.length} chunks
            </span>
            {(session.vector_queries?.length ?? 0) > 0 && (
              <div className="mt-1.5 pl-0 flex flex-col gap-0.5">
                {session.vector_queries!.map((q, i) => (
                  <div key={q.id} className="flex items-baseline gap-2">
                    <span className="text-xs text-[#4a4a4a] flex-shrink-0">Q{i + 1}</span>
                    <span className="text-xs font-mono text-[#8a8a8a] break-words" title={q.query_text}>
                      {q.query_text || `<embedding ${q.db_type}>`}
                    </span>
                    {q.collection && (
                      <span className="text-xs text-[#4a4a4a] flex-shrink-0">· {q.collection}</span>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Groundedness pill */}
          <div className="flex items-center gap-3 flex-shrink-0">
            <div className="text-right">
              <div className="text-xs text-[#6a6a6a] mb-1">Groundedness</div>
              <div className="flex items-center gap-2">
                <div className="w-32 h-2 bg-[#0f0f0f] rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all"
                    style={{ width: `${groundednessPercent}%`, backgroundColor: groundednessColor }}
                  />
                </div>
                <span className="text-sm font-bold w-10 text-right" style={{ color: groundednessColor }}>
                  {groundednessPercent}%
                </span>
              </div>
            </div>
            {hallucinated > 0 && (
              <div className="text-right">
                <div className="text-xs text-[#6a6a6a] mb-1">Hallucinated</div>
                <span className="text-sm font-bold text-[#e05c5c]">{hallucinated}/{total}</span>
              </div>
            )}
            {hallucinated > 0 && (
              <button
                onClick={async () => {
                  setAnalyzing(true);
                  setAnalyzeError(null);
                  setAnalyzeSuccess(false);
                  try {
                    const reqId = session.llm_requests?.[0]?.id;
                    await runDeepAnalysis(session.id, reqId);
                    setAnalyzeSuccess(true);
                    setTimeout(() => setAnalyzeSuccess(false), 3000);
                  } catch (e: any) {
                    setAnalyzeError(e.message ?? 'Deep analysis failed');
                  } finally {
                    setAnalyzing(false);
                  }
                }}
                disabled={analyzing}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium border transition-colors disabled:opacity-50"
                style={{
                  background: analyzeSuccess ? '#1a2a1a' : '#1a1a2a',
                  borderColor: analyzeSuccess ? '#5ce05c' : '#3a3a6a',
                  color: analyzeSuccess ? '#5ce05c' : '#a0a0e0',
                }}
                title="Run LIME perturbation attribution — makes 7 extra LLM calls to measure causal chunk influence"
              >
                {analyzing ? (
                  <>
                    <span className="inline-block w-3 h-3 border border-current border-t-transparent rounded-full animate-spin" />
                    Analyzing…
                  </>
                ) : analyzeSuccess ? (
                  <>✓ Updated</>
                ) : (
                  <>🔬 Deep Analysis</>
                )}
              </button>
            )}
            {analyzeError && (
              <div className="text-xs text-[#e05c5c] max-w-48 text-right">{analyzeError}</div>
            )}
          </div>
        </div>
      </div>

      {/* ── Two-column body ── */}
      <div className="flex flex-1 min-h-0">

        {/* LEFT — LLM output */}
        <div className="flex flex-col w-1/2 border-r border-[#2a2a2a] min-h-0">
          <div className="flex-shrink-0 px-5 pt-4 pb-2 border-b border-[#2a2a2a]">
            <h3 className="text-xs font-semibold text-[#6a6a6a] uppercase tracking-widest">
              LLM Output
            </h3>
          </div>
          <div className="flex-1 overflow-y-auto p-5">
            <div className="bg-[#1a1a1a] rounded p-4 border border-[#2a2a2a]">
              <OutputHighlighter
                tokens={attribution.output_tokens}
                tokenHeatmap={attribution.token_heatmap}
                onChunkHover={setHoveredChunkId}
              />
            </div>
          </div>
        </div>

        {/* RIGHT — Chunks / Embedding Scatter */}
        <div className="flex flex-col w-1/2 min-h-0">
          {/* Tab bar */}
          <div className="flex-shrink-0 flex items-center border-b border-[#2a2a2a] px-3 pt-1">
            <button
              onClick={() => setRightTab('chunks')}
              className={`px-3 py-2 text-xs font-medium border-b-2 transition-colors mr-1 ${
                rightTab === 'chunks'
                  ? 'border-[#e05c5c] text-[#e8a8a8]'
                  : 'border-transparent text-[#6a6a6a] hover:text-[#9a9a9a]'
              }`}
            >
              Chunks <span className="text-[#4a4a4a] font-normal ml-1">({sortedChunks.length})</span>
            </button>
            <button
              onClick={() => setRightTab('scatter')}
              className={`px-3 py-2 text-xs font-medium border-b-2 transition-colors ${
                rightTab === 'scatter'
                  ? 'border-[#5c9ae0] text-[#a8c8e8]'
                  : 'border-transparent text-[#6a6a6a] hover:text-[#9a9a9a]'
              }`}
            >
              Embedding Space
            </button>
          </div>

          {rightTab === 'chunks' ? (
            <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-2">
              {sortedChunks.map((chunk, i) => (
                <ChunkCard
                  key={chunk.chunk_id}
                  chunk={chunk}
                  rank={i + 1}
                  isHighlighted={hoveredChunkId === chunk.chunk_id}
                  onHover={setHoveredChunkId}
                />
              ))}
            </div>
          ) : (
            <div className="flex-1 min-h-0 p-2">
              <EmbeddingScatter
                sessionId={session.id}
                onPointClick={(point) => {
                  if (point.type === 'chunk') setHoveredChunkId(point.id);
                }}
                className="w-full h-full"
              />
            </div>
          )}
        </div>

      </div>
    </div>
  );
};
