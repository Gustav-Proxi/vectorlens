import React from 'react';
import { Session } from '../lib/api';

interface SessionListProps {
  sessions: Session[];
  liveSessions: string[];
  selectedSessionId: string | null;
  onSelectSession: (sessionId: string) => void;
  onNewSession: () => void;
  loading: boolean;
  error: Error | null;
  isConnecting?: boolean;
  isShowingCached?: boolean;
  onClearHistory?: () => void;
}

function formatTimeAgo(timestamp: number): string {
  const now = Date.now() / 1000;
  const diff = now - timestamp;

  if (diff < 60) return 'just now';
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

export const SessionList: React.FC<SessionListProps> = ({
  sessions,
  liveSessions,
  selectedSessionId,
  onSelectSession,
  onNewSession,
  loading,
  error,
  isConnecting = false,
  isShowingCached = false,
  onClearHistory = () => {},
}) => {
  const sortedSessions = [...sessions].sort(
    (a, b) => b.created_at - a.created_at
  );

  const liveSessObj = sortedSessions.filter((s) => liveSessions.includes(s.id));
  const historicalSessions = sortedSessions.filter(
    (s) => !liveSessions.includes(s.id)
  );

  return (
    <div className="w-60 bg-[#1a1a1a] border-r border-[#2a2a2a] flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b border-[#2a2a2a]">
        <h1 className="text-lg font-semibold text-[#e8e8e8] mb-3">VectorLens</h1>
        <button
          onClick={onNewSession}
          className="w-full bg-[#e05c5c] hover:bg-[#d04a4a] text-white font-medium py-2 px-3 rounded transition-colors text-sm"
        >
          + New Session
        </button>
      </div>

      {/* Offline Banner */}
      {isShowingCached && (
        <div className="px-3 py-1.5 border-b border-[#333] text-[#666] text-xs flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-[#555] inline-block"></span>
          Session history
        </div>
      )}

      {/* Content */}
      <div className="flex-1 overflow-y-auto flex flex-col">
        {isConnecting && !isShowingCached && (
          <div className="p-4 text-[#8a8a8a] text-xs flex items-center gap-2">
            <span className="inline-block w-2 h-2 bg-[#8a8a8a] rounded-full animate-pulse"></span>
            Connecting to VectorLens...
          </div>
        )}

        {error && !isShowingCached && (
          <div className="p-4 m-2 bg-[#2a1a1a] border border-[#8a8a8a] rounded text-[#8a8a8a] text-xs">
            Server starting • retrying...
          </div>
        )}

        {loading && !isConnecting && sessions.length === 0 && (
          <div className="p-4 text-[#8a8a8a] text-sm">Loading sessions...</div>
        )}

        {!loading && !error && !isConnecting && sessions.length === 0 && (
          <div className="p-4 text-[#8a8a8a] text-sm whitespace-pre-line">
            No sessions yet
            <br />
            Run your RAG pipeline to start tracing
          </div>
        )}

        {/* Live Sessions */}
        {liveSessObj.length > 0 && (
          <>
            <div className="px-4 pt-3 pb-2 text-[#6a6a6a] text-xs font-semibold uppercase tracking-wide">
              Live
            </div>
            <div className="divide-y divide-[#2a2a2a]">
              {liveSessObj.map((session) => (
                <button
                  key={session.id}
                  onClick={() => onSelectSession(session.id)}
                  className={`w-full text-left p-3 transition-colors ${
                    selectedSessionId === session.id
                      ? 'bg-[#2a2a2a] border-l-2 border-[#e05c5c]'
                      : 'hover:bg-[#252525]'
                  }`}
                >
                  <div className="flex items-start gap-2 mb-1">
                    <span className="inline-block w-2 h-2 bg-[#5ce05c] rounded-full mt-1 flex-shrink-0"></span>
                    <div className="truncate text-xs font-mono text-[#e8e8e8] flex-1">
                      {(session.id ?? '').substring(0, 8)}...
                    </div>
                  </div>
                  <div className="text-xs text-[#8a8a8a] mb-2 pl-4">
                    {formatTimeAgo(session.created_at)}
                  </div>
                  <div className="flex gap-2 text-xs text-[#8a8a8a] pl-4">
                    <span>{session.vector_queries_count ?? session.vector_queries?.length ?? 0} queries</span>
                    <span>{session.llm_responses_count ?? session.llm_responses?.length ?? 0} responses</span>
                  </div>
                </button>
              ))}
            </div>
          </>
        )}

        {/* Historical Sessions */}
        {historicalSessions.length > 0 && (
          <>
            <div className="px-4 pt-3 pb-2 text-[#6a6a6a] text-xs font-semibold uppercase tracking-wide">
              History
            </div>
            <div className="divide-y divide-[#2a2a2a]">
              {historicalSessions.map((session) => (
                <button
                  key={session.id}
                  onClick={() => onSelectSession(session.id)}
                  className={`w-full text-left p-3 transition-colors opacity-60 ${
                    selectedSessionId === session.id
                      ? 'bg-[#2a2a2a] border-l-2 border-[#e05c5c]'
                      : 'hover:bg-[#252525]'
                  }`}
                >
                  <div className="flex items-start gap-2 mb-1">
                    <span className="inline-block text-[#6a6a6a] mt-0.5 flex-shrink-0">📋</span>
                    <div className="truncate text-xs font-mono text-[#b8b8b8] flex-1">
                      {session.id.substring(0, 8)}...
                    </div>
                  </div>
                  <div className="text-xs text-[#8a8a8a] mb-2 pl-4">
                    {formatTimeAgo(session.created_at)}
                  </div>
                  <div className="flex gap-2 text-xs text-[#8a8a8a] pl-4">
                    <span>{session.vector_queries_count ?? session.vector_queries?.length ?? 0} queries</span>
                    <span>{session.llm_responses_count ?? session.llm_responses?.length ?? 0} responses</span>
                  </div>
                </button>
              ))}
            </div>
          </>
        )}

        {/* Spacer */}
        <div className="flex-1" />

        {/* Clear History Button */}
        {historicalSessions.length > 0 && (
          <div className="p-3 border-t border-[#2a2a2a]">
            <button
              onClick={onClearHistory}
              className="w-full text-center text-xs text-[#6a6a6a] hover:text-[#8a8a8a] transition-colors py-2"
            >
              Clear history
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
