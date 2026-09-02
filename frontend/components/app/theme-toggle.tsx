'use client';

import { useEffect, useState } from 'react';
import { useTheme } from 'next-themes';
import { MonitorIcon, MoonIcon, SunIcon } from '@phosphor-icons/react';
import { cn } from '@/lib/shadcn/utils';

interface ThemeToggleProps {
  className?: string;
}

export function ThemeToggle({ className }: ThemeToggleProps) {
  const { theme, resolvedTheme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const applyTheme = (nextTheme: 'light' | 'dark' | 'system') => {
    setTheme(nextTheme);

    const root = document.documentElement;
    const systemTheme = window.matchMedia('(prefers-color-scheme: dark)').matches
      ? 'dark'
      : 'light';
    const effectiveTheme = nextTheme === 'system' ? systemTheme : nextTheme;

    root.classList.toggle('dark', effectiveTheme === 'dark');
    root.style.colorScheme = effectiveTheme;
    window.localStorage.setItem('theme', nextTheme);
  };

  const activeTheme = mounted ? theme : 'system';
  const effectiveTheme = mounted ? resolvedTheme : undefined;

  return (
    <div
      className={cn(
        'text-foreground bg-background/90 flex w-full flex-row justify-end divide-x overflow-hidden rounded-full border shadow-lg backdrop-blur-xl',
        className
      )}
      title={effectiveTheme ? `当前：${effectiveTheme === 'dark' ? '深色' : '浅色'}` : undefined}
    >
      <span className="sr-only">颜色模式切换</span>
      <button
        type="button"
        onClick={() => applyTheme('dark')}
        className="cursor-pointer p-1 pl-1.5"
      >
        <span className="sr-only">启用深色模式</span>
        <MoonIcon
          suppressHydrationWarning
          size={16}
          weight="bold"
          className={cn(activeTheme !== 'dark' && 'opacity-25')}
        />
      </button>
      <button
        type="button"
        onClick={() => applyTheme('light')}
        className="cursor-pointer px-1.5 py-1"
      >
        <span className="sr-only">启用浅色模式</span>
        <SunIcon
          suppressHydrationWarning
          size={16}
          weight="bold"
          className={cn(activeTheme !== 'light' && 'opacity-25')}
        />
      </button>
      <button
        type="button"
        onClick={() => applyTheme('system')}
        className="cursor-pointer p-1 pr-1.5"
      >
        <span className="sr-only">跟随系统颜色模式</span>
        <MonitorIcon
          suppressHydrationWarning
          size={16}
          weight="bold"
          className={cn(activeTheme !== 'system' && 'opacity-25')}
        />
      </button>
    </div>
  );
}
