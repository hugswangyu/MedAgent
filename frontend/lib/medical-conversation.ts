export type ConsultationInputMode = 'text' | 'voice';
export type ConsultationChannel = 'text' | 'voice';
export type ConsultationRole = 'user' | 'assistant';

export interface ConsultationEvidence {
  id: string;
  title: string;
  preview?: string;
  score?: number;
}

export interface ConsultationMessage {
  id: string;
  role: ConsultationRole;
  channel: ConsultationChannel;
  content: string;
  createdAt: string;
  evidence?: ConsultationEvidence[];
  pending?: boolean;
}

export interface ConversationIdentity {
  conversationId: string;
  textSessionId: string;
  voiceClientId: string;
}

export function createConversationIdentity(uuid: string): ConversationIdentity {
  const normalized = uuid.replace(/[^a-zA-Z0-9]/g, '').slice(0, 48);
  const conversationId = `conv_${normalized}`;
  return {
    conversationId,
    textSessionId: `web_${conversationId}`,
    voiceClientId: conversationId,
  };
}

export function switchConsultationMode<T extends { inputMode: ConsultationInputMode }>(
  state: T,
  inputMode: ConsultationInputMode
): T {
  return { ...state, inputMode };
}

function messageKey(message: ConsultationMessage) {
  return `${message.channel}:${message.role}:${message.content.trim().replace(/\s+/g, ' ')}`;
}

/** Joins text turns, persisted voice turns and in-flight transcriptions. */
export function mergeConversationTimeline(
  textMessages: ConsultationMessage[],
  voiceMessages: ConsultationMessage[]
): ConsultationMessage[] {
  const merged = new Map<string, ConsultationMessage>();
  for (const message of [...textMessages, ...voiceMessages]) {
    const key = messageKey(message);
    const current = merged.get(key);
    if (!current) {
      merged.set(key, message);
      continue;
    }
    const currentEvidence = current.evidence?.length ?? 0;
    const nextEvidence = message.evidence?.length ?? 0;
    if (nextEvidence > currentEvidence || (current.pending && !message.pending)) {
      merged.set(key, message);
    }
  }
  return [...merged.values()].sort((a, b) => {
    const timeDelta = Date.parse(a.createdAt) - Date.parse(b.createdAt);
    return timeDelta || a.id.localeCompare(b.id);
  });
}

export type VoiceConnectionState =
  | 'disconnected'
  | 'connecting'
  | 'listening'
  | 'thinking'
  | 'speaking';

export function resolveVoiceConnectionState(
  agentState: string | undefined,
  isConnected: boolean,
  isStarting: boolean
): VoiceConnectionState {
  if (isStarting || agentState === 'connecting' || agentState === 'initializing') {
    return 'connecting';
  }
  if (!isConnected) return 'disconnected';
  if (agentState === 'thinking') return 'thinking';
  if (agentState === 'speaking') return 'speaking';
  return 'listening';
}

export const VOICE_STATE_LABELS: Record<VoiceConnectionState, string> = {
  disconnected: '语音已断开',
  connecting: '正在连接语音',
  listening: '正在聆听',
  thinking: '正在思考',
  speaking: '正在播报',
};
