import { cn } from '@/lib/shadcn/utils';

type MedAgentLogoProps = {
  className?: string;
  title?: string;
};

/** MedAgent's medical-cross mark, carried forward from the legacy frontend. */
export function MedAgentLogo({ className, title }: MedAgentLogoProps) {
  return (
    <svg
      viewBox="0 0 24 24"
      role={title ? 'img' : undefined}
      aria-hidden={title ? undefined : true}
      aria-label={title}
      className={cn('shrink-0', className)}
      xmlns="http://www.w3.org/2000/svg"
    >
      <rect width="24" height="24" rx="6" fill="currentColor" />
      <rect x="9" y="4" width="6" height="16" rx="1.8" fill="white" />
      <rect x="4" y="9" width="16" height="6" rx="1.8" fill="white" />
    </svg>
  );
}
