import { useState } from 'react';
import { SessionList } from './components/SessionList';
import { AttributionView } from './components/AttributionView';
import { useSession, useSessions } from './hooks/useSession';
import { newSession } from './lib/api';

function App() {
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(
    null
  );
  const {
    sessions,
    liveSessions,
    loading: sessionsLoading,
    error: sessionsError,
    isConnecting,
    isShowingCached,
    refetch,
    clearHistory,
  } = useSessions();

  // Determine if selected session is live
  const isLiveSession = selectedSessionId ? liveSessions.includes(selectedSessionId) : true;

  const {
    session,
    loading: sessionLoading,
    error: sessionError,
    isCached,
  } = useSession(selectedSessionId, isLiveSession);

  const handleNewSession = async () => {
    try {
      const newSess = await newSession();
      setSelectedSessionId(newSess.id);
      refetch();
    } catch (error) {
      console.error('Error creating new session:', error);
    }
  };

  return (
    <div className="flex h-screen bg-[#0f0f0f]">
      {/* Sidebar */}
      <SessionList
        sessions={sessions}
        liveSessions={liveSessions}
        selectedSessionId={selectedSessionId}
        onSelectSession={setSelectedSessionId}
        onNewSession={handleNewSession}
        loading={sessionsLoading}
        error={sessionsError}
        isConnecting={isConnecting}
        isShowingCached={isShowingCached}
        onClearHistory={clearHistory}
      />

      {/* Main Content */}
      <div className="flex-1 flex">
        {selectedSessionId ? (
          <AttributionView
            session={session}
            loading={sessionLoading}
            error={sessionError}
            isCached={isCached}
          />
        ) : (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center">
              <h2 className="text-2xl font-semibold text-[#e8e8e8] mb-2">
                Welcome to VectorLens
              </h2>
              <p className="text-[#8a8a8a] mb-6">
                Select a session or create a new one to get started
              </p>
              {sessions.length === 0 && (
                <button
                  onClick={handleNewSession}
                  className="bg-[#e05c5c] hover:bg-[#d04a4a] text-white font-medium py-2 px-6 rounded transition-colors"
                >
                  Create Your First Session
                </button>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
