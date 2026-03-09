export interface RetrievedChunk {
  chunk_id: string;
  text: string;
  score: number;
  attribution_score: number;
  caused_hallucination: boolean;
  metadata: Record<string, unknown>;
}

export interface OutputToken {
  text: string;
  position: number;
  is_hallucinated: boolean;
  hallucination_score: number;
  chunk_attributions: Record<string, number>;
}

export interface TokenHeatmapEntry {
  text: string;
  position: number;
  chunk_attributions: Record<string, number>;
}

export interface AttributionResult {
  id: string;
  session_id: string;
  chunks: RetrievedChunk[];
  output_tokens: OutputToken[];
  overall_groundedness: number;
  hallucinated_spans: [number, number][];
  token_heatmap?: TokenHeatmapEntry[];
}

export interface VectorQuery {
  id: string;
  query: string;
  timestamp: number;
  vector_model: string;
}

export interface LLMRequest {
  id: string;
  prompt: string;
  timestamp: number;
  model: string;
}

export interface LLMResponse {
  id: string;
  request_id: string;
  text: string;
  timestamp: number;
  tokens: number;
}

export interface Session {
  id: string;
  created_at: number;
  // Full detail (from GET /api/sessions/{id})
  vector_queries?: VectorQuery[];
  llm_requests?: LLMRequest[];
  llm_responses?: LLMResponse[];
  attributions?: AttributionResult[];
  // Summary counts (from GET /api/sessions list)
  vector_queries_count?: number;
  llm_requests_count?: number;
  llm_responses_count?: number;
  attributions_count?: number;
}

// Use the same host/port the dashboard was served from — avoids localhost vs 127.0.0.1 mismatch
// Relative base — works regardless of hostname/port (no localhost vs 127.0.0.1 issues)
export const API_BASE = '/api';
export const WS_BASE = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}`;

export async function fetchSessions(): Promise<Session[]> {
  try {
    const response = await fetch(`${API_BASE}/sessions`);
    if (!response.ok) {
      throw new Error(`Failed to fetch sessions: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching sessions:', error);
    throw error;
  }
}

export async function fetchSession(id: string): Promise<Session> {
  try {
    const response = await fetch(`${API_BASE}/sessions/${id}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch session: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error(`Error fetching session ${id}:`, error);
    throw error;
  }
}

export async function newSession(): Promise<Session> {
  try {
    const response = await fetch(`${API_BASE}/sessions/new`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });
    if (!response.ok) {
      throw new Error(`Failed to create session: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error creating session:', error);
    throw error;
  }
}
