import { useState, useEffect, useRef } from 'react';
import { Session, fetchSession, API_BASE, WS_BASE } from '../lib/api';
import {
  getStoredSession,
  loadStoredSessions as loadStoredSessionsUtil,
  saveSessions as saveSessionsUtil,
  markSessionsLive as markSessionsLiveUtil,
  clearStoredSessions as clearStoredSessionsUtil,
} from '../lib/storage';

interface UseSessionReturn {
  session: Session | null;
  loading: boolean;
  error: Error | null;
  isCached: boolean;
}

export function useSession(
  sessionId: string | null,
  isLiveSession: boolean = true
): UseSessionReturn {
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [isCached, setIsCached] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (!sessionId) {
      setSession(null);
      setIsCached(false);
      return;
    }

    // If not a live session, load from localStorage
    if (!isLiveSession) {
      const stored = getStoredSession(sessionId);
      if (stored) {
        // Remove storage metadata before setting
        const { _stored_at, _is_live, ...sessionData } = stored;
        setSession(sessionData as Session);
        setIsCached(true);
        setLoading(false);
        setError(null);
      } else {
        setError(new Error('Session not found'));
        setLoading(false);
      }
      return;
    }

    // For live sessions, fetch from server
    setLoading(true);
    setError(null);
    setIsCached(false);

    // Fetch initial session data
    fetchSession(sessionId)
      .then((data) => {
        setSession(data);
        setIsCached(false);
      })
      .catch((err) => {
        // Try to load from localStorage as fallback
        const stored = getStoredSession(sessionId);
        if (stored) {
          const { _stored_at, _is_live, ...sessionData } = stored;
          setSession(sessionData as Session);
          setIsCached(true);
          setError(null);
        } else {
          setError(err instanceof Error ? err : new Error(String(err)));
        }
      })
      .finally(() => setLoading(false));

    // Connect to WebSocket
    const connectWebSocket = () => {
      try {
        const wsUrl = `${WS_BASE}/ws?session_id=${sessionId}`;
        const ws = new WebSocket(wsUrl);

        ws.onopen = () => {
          console.log(`[WS] Connected for session ${sessionId}`);
          if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
          }
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            console.log('[WS] Message:', data);
            // Re-fetch session to get latest state
            fetchSession(sessionId)
              .then((data) => {
                setSession(data);
                setIsCached(false);
              })
              .catch((err) => console.error('Error refetching session:', err));
          } catch (err) {
            console.error('Error parsing WebSocket message:', err);
          }
        };

        ws.onerror = (event) => {
          console.error('[WS] Error:', event);
        };

        ws.onclose = () => {
          console.log('[WS] Disconnected, reconnecting in 2s...');
          wsRef.current = null;
          reconnectTimeoutRef.current = setTimeout(() => {
            connectWebSocket();
          }, 2000);
        };

        wsRef.current = ws;
      } catch (err) {
        console.error('Error connecting to WebSocket:', err);
        reconnectTimeoutRef.current = setTimeout(() => {
          connectWebSocket();
        }, 2000);
      }
    };

    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [sessionId, isLiveSession]);

  return { session, loading, error, isCached };
}

interface UseSessionsReturn {
  sessions: Session[];
  liveSessions: string[];
  loading: boolean;
  error: Error | null;
  isConnecting: boolean;
  isShowingCached: boolean;
  refetch: () => void;
  clearHistory: () => void;
}

export function useSessions(): UseSessionsReturn {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [liveSessions, setLiveSessions] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [isConnecting, setIsConnecting] = useState(true);
  const [isShowingCached, setIsShowingCached] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pollIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const firstAttemptTimeRef = useRef<number | null>(null);
  const connectionAttemptsRef = useRef<number>(0);
  const wsAttemptsRef = useRef<number>(0);

  const fetchSessions = async () => {
    try {
      const response = await fetch(`${API_BASE}/sessions`);
      if (!response.ok) {
        throw new Error(`Failed to fetch sessions: ${response.statusText}`);
      }
      const data = await response.json();
      const serverSessions = Array.isArray(data) ? data : [];

      // Load stored sessions and merge
      const storedSessions = loadStoredSessionsUtil();
      const sessionMap = new Map<string, Session>();

      // Add stored sessions first
      storedSessions.forEach((s) => {
        const { _stored_at, _is_live, ...session } = s;
        sessionMap.set(s.id, session as Session);
      });

      // Override with server sessions (server is authoritative)
      serverSessions.forEach((s: Session) => {
        sessionMap.set(s.id, s);
      });

      const merged = Array.from(sessionMap.values()).sort(
        (a, b) => b.created_at - a.created_at
      );

      setSessions(merged);
      setLiveSessions(serverSessions.map((s: Session) => s.id));

      // Save merged sessions and mark live ones
      saveSessionsUtil(serverSessions);
      markSessionsLiveUtil(new Set(serverSessions.map((s: Session) => s.id)));

      setError(null);
      setIsConnecting(false);
      setIsShowingCached(false);
      return true;
    } catch (err) {
      const now = Date.now();

      // Track first attempt time
      if (firstAttemptTimeRef.current === null) {
        firstAttemptTimeRef.current = now;
      }

      connectionAttemptsRef.current += 1;
      const timeSinceFirstAttempt = now - firstAttemptTimeRef.current;

      // Load from localStorage as fallback
      const stored = loadStoredSessionsUtil();
      if (stored.length > 0) {
        const sessions = stored.map((s) => {
          const { _stored_at, _is_live, ...session } = s;
          return session as Session;
        });
        setSessions(sessions);
        setIsShowingCached(true);
        setError(null);
        setIsConnecting(false);
        setLoading(false);
        return false;
      }

      // Only show error after 30s of continuous failure
      if (timeSinceFirstAttempt > 30000) {
        setError(err instanceof Error ? err : new Error(String(err)));
        setIsConnecting(false);
      } else {
        // Keep trying silently
        setError(null);
        setIsConnecting(true);
      }

      // Only set loading false after first 3s
      if (timeSinceFirstAttempt > 3000) {
        setLoading(false);
      }

      return false;
    }
  };

  useEffect(() => {
    // Load from localStorage immediately for instant display
    const stored = loadStoredSessionsUtil();
    if (stored.length > 0) {
      const sessions = stored.map((s) => {
        const { _stored_at, _is_live, ...session } = s;
        return session as Session;
      });
      setSessions(sessions);
      setLoading(false);
    }

    // Reset attempt tracking
    firstAttemptTimeRef.current = null;
    connectionAttemptsRef.current = 0;
    wsAttemptsRef.current = 0;
    if (stored.length === 0) {
      setLoading(true);
    }
    setIsConnecting(true);

    // Fetch from server in background
    fetchSessions();

    // Setup polling - fetch every 2s when connected, 1s when trying to connect
    const startPolling = () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }

      pollIntervalRef.current = setInterval(async () => {
        await fetchSessions();
      }, isConnecting ? 1000 : 2000);
    };

    startPolling();

    // Connect to global WebSocket for updates with exponential backoff
    let wsBackoffMs = 1000;
    const maxBackoffMs = 15000;

    const connectWebSocket = () => {
      try {
        const wsUrl = `${WS_BASE}/ws`;
        const ws = new WebSocket(wsUrl);

        ws.onopen = () => {
          console.log('[WS] Connected to global session updates');
          wsAttemptsRef.current = 0;
          wsBackoffMs = 1000;
          setIsConnecting(false);
          if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
          }
        };

        ws.onmessage = () => {
          console.log('[WS] Session update, refetching...');
          fetchSessions();
        };

        ws.onerror = (event) => {
          console.error('[WS] Error:', event);
        };

        ws.onclose = () => {
          console.log('[WS] Disconnected');
          wsRef.current = null;
          wsAttemptsRef.current += 1;

          // Exponential backoff: 1s, 2s, 4s, 8s, max 15s
          wsBackoffMs = Math.min(
            1000 * Math.pow(2, wsAttemptsRef.current - 1),
            maxBackoffMs
          );
          console.log(`[WS] Reconnecting in ${wsBackoffMs}ms...`);

          reconnectTimeoutRef.current = setTimeout(() => {
            connectWebSocket();
          }, wsBackoffMs);
        };

        wsRef.current = ws;
      } catch (err) {
        console.error('Error connecting to WebSocket:', err);
        wsAttemptsRef.current += 1;
        wsBackoffMs = Math.min(
          1000 * Math.pow(2, wsAttemptsRef.current - 1),
          maxBackoffMs
        );

        reconnectTimeoutRef.current = setTimeout(() => {
          connectWebSocket();
        }, wsBackoffMs);
      }
    };

    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, []);

  return {
    sessions,
    liveSessions,
    loading,
    error,
    isConnecting,
    isShowingCached,
    refetch: fetchSessions,
    clearHistory: clearStoredSessionsUtil,
  };
}
