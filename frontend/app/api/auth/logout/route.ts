import { NextResponse } from 'next/server';
import { ACCESS_COOKIE } from '@/lib/server-proxy';

export async function POST() {
  const response = NextResponse.json({ status: 'ok' });
  response.cookies.set(ACCESS_COOKIE, '', {
    httpOnly: true,
    sameSite: 'lax',
    secure: process.env.NODE_ENV === 'production',
    path: '/',
    maxAge: 0,
  });
  return response;
}
