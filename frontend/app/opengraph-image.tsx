import { headers } from 'next/headers';
import { ImageResponse } from 'next/og';
import { getAppConfig } from '@/lib/utils';

export const alt = 'MedAgent 智能医疗助手';
export const size = { width: 1200, height: 630 };
export const contentType = 'image/png';

export default async function Image() {
  const appConfig = await getAppConfig(await headers());

  return new ImageResponse(
    (
      <div
        style={{
          display: 'flex',
          width: '100%',
          height: '100%',
          alignItems: 'center',
          background: 'linear-gradient(135deg, #183c38 0%, #285b56 58%, #e8f3f1 58%)',
          color: 'white',
          padding: '72px 84px',
        }}
      >
        <div style={{ display: 'flex', flexDirection: 'column', width: 700 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 22 }}>
            <div
              style={{
                display: 'flex',
                position: 'relative',
                width: 76,
                height: 76,
                borderRadius: 18,
                background: '#e7f5f2',
              }}
            >
              <div
                style={{
                  position: 'absolute',
                  left: 29,
                  top: 12,
                  width: 18,
                  height: 52,
                  borderRadius: 5,
                  background: '#183c38',
                }}
              />
              <div
                style={{
                  position: 'absolute',
                  left: 12,
                  top: 29,
                  width: 52,
                  height: 18,
                  borderRadius: 5,
                  background: '#183c38',
                }}
              />
            </div>
            <div style={{ display: 'flex', fontSize: 42, fontWeight: 700 }}>MedAgent</div>
          </div>
          <div style={{ display: 'flex', marginTop: 80, fontSize: 58, fontWeight: 700 }}>
            {appConfig.pageTitle}
          </div>
          <div style={{ display: 'flex', marginTop: 24, fontSize: 27, color: '#cfe2df' }}>
            {appConfig.pageDescription}
          </div>
        </div>
        <div
          style={{
            display: 'flex',
            marginLeft: 'auto',
            width: 250,
            height: 250,
            alignItems: 'center',
            justifyContent: 'center',
            borderRadius: 64,
            background: '#ffffff',
            boxShadow: '0 28px 70px rgba(24, 60, 56, 0.18)',
          }}
        >
          <div style={{ display: 'flex', position: 'relative', width: 150, height: 150 }}>
            <div
              style={{
                position: 'absolute',
                left: 57,
                width: 36,
                height: 150,
                borderRadius: 12,
                background: '#183c38',
              }}
            />
            <div
              style={{
                position: 'absolute',
                top: 57,
                width: 150,
                height: 36,
                borderRadius: 12,
                background: '#183c38',
              }}
            />
          </div>
        </div>
      </div>
    ),
    size
  );
}
