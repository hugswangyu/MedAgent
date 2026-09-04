import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  // Next standalone tracing relies on symlinks that standard Windows accounts
  // cannot create. Linux/container deployments keep the existing standalone output.
  output: process.platform === 'win32' ? undefined : 'standalone',
};

export default nextConfig;
