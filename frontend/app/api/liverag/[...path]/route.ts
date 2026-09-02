import { type NextRequest } from 'next/server';
import { proxyRequest } from '@/lib/server-proxy';

const MEDLIVE_API_BASE =
  process.env.MEDLIVE_API_BASE ?? process.env.LIVERAG_API_BASE ?? 'http://127.0.0.1:9821';

interface RouteContext {
  params: Promise<{ path?: string[] }>;
}

async function proxy(request: NextRequest, context: RouteContext) {
  const { path = [] } = await context.params;
  return proxyRequest(request, MEDLIVE_API_BASE, path);
}

export const GET = proxy;
export const POST = proxy;
export const PUT = proxy;
export const PATCH = proxy;
export const DELETE = proxy;
