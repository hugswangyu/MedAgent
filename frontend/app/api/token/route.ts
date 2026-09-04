import { type NextRequest, NextResponse } from 'next/server';
import { ACCESS_COOKIE, VOICE_SESSION_COOKIE } from '@/lib/server-proxy';

const MEDLIVE_API_BASE =
  process.env.MEDLIVE_API_BASE ?? process.env.LIVERAG_API_BASE ?? 'http://127.0.0.1:9821';

interface Envelope<T> {
  status: 'ok' | 'error';
  data?: T;
  error?: { message?: string };
}

interface KnowledgeState {
  configured?: { kb_id?: string } | null;
  active_session?: { kb_id?: string } | null;
}

interface VoiceSession {
  session_id: string;
  room_name?: string;
  identity?: string;
  livekit_url?: string;
  token: string;
  livekit?: {
    room_name?: string;
    participant_identity?: string;
    livekit_url?: string;
  };
}

async function liveragFetch<T>(path: string, token: string, init?: RequestInit): Promise<T> {
  const response = await fetch(new URL(path, MEDLIVE_API_BASE), {
    ...init,
    headers: {
      authorization: 'Bearer ' + token,
      'content-type': 'application/json',
      ...(init?.headers ?? {}),
    },
    cache: 'no-store',
  });
  const envelope = (await response.json()) as Envelope<T>;
  if (!response.ok || envelope.status !== 'ok' || !envelope.data) {
    throw new Error(envelope.error?.message ?? 'MedLive voice session request failed');
  }
  return envelope.data;
}

export const revalidate = 0;

export async function POST(request: NextRequest) {
  const token = request.cookies.get(ACCESS_COOKIE)?.value;
  if (!token) {
    return NextResponse.json({ detail: 'authentication required' }, { status: 401 });
  }

  try {
    const knowledge = await liveragFetch<KnowledgeState>('/session/knowledge-base', token);
    const kbId = knowledge.active_session?.kb_id ?? knowledge.configured?.kb_id;
    if (!kbId) {
      return NextResponse.json({ detail: '请先选择一组可用的病历或医学资料' }, { status: 409 });
    }
    const requestedConversationId = request.nextUrl.searchParams.get('conversation_id') ?? '';
    const conversationId = /^conv_[a-zA-Z0-9]{8,48}$/.test(requestedConversationId)
      ? requestedConversationId
      : `conv_${crypto.randomUUID().replaceAll('-', '')}`;
    const session = await liveragFetch<VoiceSession>('/voice/sessions', token, {
      method: 'POST',
      body: JSON.stringify({
        kb_id: kbId,
        // MedLive owns its session_id. client_id safely correlates it with the
        // parent medical conversation shown by this web client.
        client_id: conversationId,
        client_type: 'web',
      }),
    });

    const response = NextResponse.json(
      {
        serverUrl: session.livekit_url ?? session.livekit?.livekit_url,
        roomName: session.room_name ?? session.livekit?.room_name,
        participantName: session.identity ?? session.livekit?.participant_identity ?? 'user',
        participantToken: session.token,
        sessionId: session.session_id,
        conversationId,
      },
      { headers: { 'cache-control': 'no-store' } }
    );
    response.cookies.set(VOICE_SESSION_COOKIE, session.session_id, {
      httpOnly: true,
      sameSite: 'lax',
      secure: process.env.NODE_ENV === 'production',
      path: '/',
      maxAge: 60 * 60 * 8,
    });
    return response;
  } catch (error) {
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : 'voice session unavailable' },
      { status: 502 }
    );
  }
}
