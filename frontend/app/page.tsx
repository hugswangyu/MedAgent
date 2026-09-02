import { headers } from 'next/headers';
import { UnifiedApp } from '@/components/app/unified-app';
import { getAppConfig } from '@/lib/utils';

export default async function Page() {
  const hdrs = await headers();
  const appConfig = await getAppConfig(hdrs);
  return <UnifiedApp appConfig={appConfig} />;
}
