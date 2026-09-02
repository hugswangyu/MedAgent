import { type NextRequest, NextResponse } from 'next/server';
import { VOICE_SESSION_COOKIE, proxyRequest } from '@/lib/server-proxy';

const MEDLIVE_API_BASE =
  process.env.MEDLIVE_API_BASE ?? process.env.LIVERAG_API_BASE ?? 'http://127.0.0.1:9821';

export const revalidate = 0;

export async function GET(request: NextRequest) {
  const sessionId = request.cookies.get(VOICE_SESSION_COOKIE)?.value;
  if (!sessionId) {
    return NextResponse.json({ detail: 'current voice session not found' }, { status: 404 });
  }
  return proxyRequest(request, MEDLIVE_API_BASE, [
    'voice',
    'sessions',
    encodeURIComponent(sessionId),
    'rag-context',
  ]);
}
