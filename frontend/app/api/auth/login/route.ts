import { type NextRequest, NextResponse } from 'next/server';
import { ACCESS_COOKIE } from '@/lib/server-proxy';

const MEDAGENT_API_BASE = process.env.MEDAGENT_API_BASE ?? 'http://127.0.0.1:8000';

export async function POST(request: NextRequest) {
  const response = await fetch(new URL('/auth/login', MEDAGENT_API_BASE), {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: await request.text(),
    cache: 'no-store',
  });
  const payload = (await response.json()) as {
    access_token?: unknown;
    user_id?: unknown;
    username?: unknown;
    detail?: unknown;
  };
  if (!response.ok) {
    return NextResponse.json(
      {
        detail: typeof payload.detail === 'string' ? payload.detail : 'login failed',
      },
      { status: response.status }
    );
  }
  if (
    typeof payload.access_token !== 'string' ||
    typeof payload.user_id !== 'string' ||
    typeof payload.username !== 'string'
  ) {
    return NextResponse.json({ detail: 'invalid login response' }, { status: 502 });
  }

  const result = NextResponse.json({
    user_id: payload.user_id,
    username: payload.username,
  });
  result.cookies.set(ACCESS_COOKIE, payload.access_token, {
    httpOnly: true,
    sameSite: 'lax',
    secure: process.env.NODE_ENV === 'production',
    path: '/',
    maxAge: 60 * 60 * 8,
  });
  return result;
}
