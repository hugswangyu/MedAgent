export interface MedAgentUser {
  user_id: string;
  username: string;
  is_admin?: boolean;
}

export interface MedicalMemory {
  memory_id: string;
  memory_type: string;
  content: string;
  status: 'proposed' | 'confirmed' | 'superseded' | 'rejected';
  confidence?: number;
  created_at?: string;
  updated_at?: string;
}

async function errorMessage(response: Response) {
  try {
    const payload = (await response.json()) as { detail?: string };
    return payload.detail ?? response.statusText;
  } catch {
    return response.statusText;
  }
}

export async function medagentJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch('/api/medagent' + path, {
    ...init,
    headers: {
      ...(init?.body ? { 'content-type': 'application/json' } : {}),
      ...(init?.headers ?? {}),
    },
    cache: 'no-store',
  });
  if (!response.ok) throw new Error(await errorMessage(response));
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export async function login(username: string, password: string) {
  const response = await fetch('/api/auth/login', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ username, password }),
  });
  if (!response.ok) throw new Error(await errorMessage(response));
  return (await response.json()) as {
    user_id: string;
    username: string;
  };
}

export async function logout() {
  await fetch('/api/auth/logout', { method: 'POST' });
}

export function getCurrentUser() {
  return medagentJson<MedAgentUser>('/auth/me');
}

export async function listMemories() {
  return medagentJson<{ memories: MedicalMemory[]; total: number }>('/memories');
}

export function transitionMemory(memoryId: string, action: 'confirm' | 'reject') {
  return medagentJson<MedicalMemory>('/memories/' + encodeURIComponent(memoryId) + '/' + action, {
    method: 'POST',
  });
}

export function correctMemory(memoryId: string, content: string) {
  return medagentJson<MedicalMemory>('/memories/' + encodeURIComponent(memoryId) + '/correct', {
    method: 'POST',
    body: JSON.stringify({ content, structured_value: {}, confidence: 1 }),
  });
}

export function deleteMemory(memoryId: string) {
  return medagentJson<void>('/memories/' + encodeURIComponent(memoryId), { method: 'DELETE' });
}

export interface ChatEvent {
  type?: string;
  content?: string;
  [key: string]: unknown;
}

export async function streamTextChat(
  payload: {
    message: string;
    session_id: string;
    knowledge_base?: string;
    provider?: string;
    model?: string;
  },
  onEvent: (event: ChatEvent) => void,
  signal?: AbortSignal
) {
  const response = await fetch('/api/medagent/chat/stream', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  });
  if (!response.ok || !response.body) throw new Error(await errorMessage(response));

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  for (;;) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
    const frames = buffer.split('\n\n');
    buffer = frames.pop() ?? '';
    for (const frame of frames) {
      const data = frame
        .split('\n')
        .filter((line) => line.startsWith('data:'))
        .map((line) => line.slice(5).trim())
        .join('\n');
      if (!data || data === '[DONE]') continue;
      try {
        onEvent(JSON.parse(data) as ChatEvent);
      } catch {
        onEvent({ type: 'content', content: data });
      }
    }
    if (done) break;
  }
}

export interface SessionSummary {
  session_id: string;
  message_count: number;
  updated_at: string;
}

export interface SessionDetail {
  session_id: string;
  messages: Array<{
    type: 'human' | 'ai';
    content: string;
    rag_trace?: Record<string, unknown> | null;
  }>;
}

export function listSessions() {
  return medagentJson<{ sessions: SessionSummary[] }>('/sessions');
}

export function getSession(sessionId: string) {
  return medagentJson<SessionDetail>('/sessions/' + encodeURIComponent(sessionId));
}

export async function register(username: string, password: string) {
  const response = await fetch('/api/auth/register', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ username, password }),
  });
  if (!response.ok) throw new Error(await errorMessage(response));
  return (await response.json()) as {
    user_id: string;
    username: string;
  };
}
