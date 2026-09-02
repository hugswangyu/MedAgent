import { type NextRequest, NextResponse } from 'next/server';

export const ACCESS_COOKIE = 'medagent_access_token';
export const VOICE_SESSION_COOKIE = 'medagent_voice_session_id';

export async function proxyRequest(
  request: NextRequest,
  upstreamBase: string,
  path: string[],
  options: { injectAuth?: boolean } = {}
) {
  const upstreamUrl = new URL('/' + path.join('/'), upstreamBase);
  upstreamUrl.search = request.nextUrl.search;

  const headers = new Headers(request.headers);
  headers.delete('host');
  headers.delete('connection');
  headers.delete('cookie');
  if (options.injectAuth !== false) {
    const token = request.cookies.get(ACCESS_COOKIE)?.value;
    if (!token) {
      return NextResponse.json({ detail: 'authentication required' }, { status: 401 });
    }
    headers.set('authorization', 'Bearer ' + token);
  }

  const response = await fetch(upstreamUrl, {
    method: request.method,
    headers,
    body:
      request.method === 'GET' || request.method === 'HEAD'
        ? undefined
        : await request.arrayBuffer(),
    cache: 'no-store',
  });
  const responseHeaders = new Headers(response.headers);
  responseHeaders.delete('content-encoding');
  responseHeaders.delete('content-length');
  responseHeaders.delete('set-cookie');

  return new NextResponse(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers: responseHeaders,
  });
}
