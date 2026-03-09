import { Session } from './api';

const STORAGE_KEY = 'vectorlens_sessions';
const MAX_STORED_SESSIONS = 50;

export interface StoredSession extends Session {
  _stored_at: number;
  _is_live: boolean;
}

export function loadStoredSessions(): StoredSession[] {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) return [];

    const sessions = JSON.parse(stored) as StoredSession[];
    // Sort by created_at descending
    return sessions.sort((a, b) => b.created_at - a.created_at);
  } catch (err) {
    console.error('Error loading stored sessions:', err);
    return [];
  }
}

export function saveSessions(sessions: Session[]): void {
  try {
    // Load existing stored sessions
    const existing = loadStoredSessions();

    // Mark all as not live before merging
    const sessionsToStore = sessions.map((s) => ({
      ...s,
      _stored_at: Date.now(),
      _is_live: false,
    } as StoredSession));

    // Merge: server sessions override stored, keep historical ones
    const sessionMap = new Map<string, StoredSession>();

    // First add all existing stored sessions
    existing.forEach((s) => {
      sessionMap.set(s.id, s);
    });

    // Then override/add server sessions
    sessionsToStore.forEach((s) => {
      sessionMap.set(s.id, s);
    });

    // Keep only MAX_STORED_SESSIONS most recent
    const merged = Array.from(sessionMap.values())
      .sort((a, b) => b.created_at - a.created_at)
      .slice(0, MAX_STORED_SESSIONS);

    // Size guard before storing
    const serialized = JSON.stringify(merged);
    const SIZE_LIMIT = 4 * 1024 * 1024; // 4MB — leave 1MB buffer from 5MB limit

    if (serialized.length > SIZE_LIMIT) {
      // Too large — store summaries only (strip vector_queries and llm_responses content)
      const summaries = merged.map(s => ({
        ...s,
        // Keep metadata but strip large arrays for storage
        vector_queries: [] as any[],
        llm_requests: [] as any[],
        llm_responses: [] as any[],
        attributions: [] as any[],
      }));
      const summaryStr = JSON.stringify(summaries);
      if (summaryStr.length <= SIZE_LIMIT) {
        localStorage.setItem(STORAGE_KEY, summaryStr);
        console.warn('VectorLens: sessions too large for localStorage, storing summaries only');
        return;
      }
      // Even summaries too large — skip storage entirely
      console.warn('VectorLens: sessions too large to cache, skipping localStorage');
      return;
    }

    localStorage.setItem(STORAGE_KEY, serialized);
  } catch (err) {
    if (err instanceof Error && err.name === 'QuotaExceededError') {
      console.warn('localStorage quota exceeded, removing oldest sessions');
      // Remove oldest sessions and retry
      const existing = loadStoredSessions();
      const trimmed = existing.slice(0, Math.max(1, existing.length - 10));
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(trimmed));
      } catch (retryErr) {
        console.error('Failed to save sessions after trimming:', retryErr);
      }
    } else {
      console.error('Error saving sessions:', err);
    }
  }
}

export function markSessionsLive(liveIds: Set<string>): void {
  try {
    const stored = loadStoredSessions();
    const updated = stored.map((s) => ({
      ...s,
      _is_live: liveIds.has(s.id),
    }));
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
  } catch (err) {
    console.error('Error marking sessions as live:', err);
  }
}

export function clearStoredSessions(): void {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch (err) {
    console.error('Error clearing stored sessions:', err);
  }
}

export function getStoredSession(sessionId: string): StoredSession | null {
  try {
    const sessions = loadStoredSessions();
    return sessions.find((s) => s.id === sessionId) || null;
  } catch (err) {
    console.error('Error retrieving stored session:', err);
    return null;
  }
}
