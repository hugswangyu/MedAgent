import { type NextRequest } from 'next/server';
import { proxyRequest } from '@/lib/server-proxy';

const MEDAGENT_API_BASE = process.env.MEDAGENT_API_BASE ?? 'http://127.0.0.1:8000';

interface RouteContext {
  params: Promise<{ path?: string[] }>;
}

async function proxy(request: NextRequest, context: RouteContext) {
  const { path = [] } = await context.params;
  return proxyRequest(request, MEDAGENT_API_BASE, path);
}

export const GET = proxy;
export const POST = proxy;
export const PUT = proxy;
export const PATCH = proxy;
export const DELETE = proxy;
