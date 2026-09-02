'use client';

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { DatabaseIcon } from 'lucide-react';
import { AnimatePresence, type MotionProps, motion } from 'motion/react';
import { useAgent, useSessionContext, useSessionMessages } from '@livekit/components-react';
import { AgentChatTranscript } from '@/components/agents-ui/agent-chat-transcript';
import {
  AgentControlBar,
  type AgentControlBarControls,
} from '@/components/agents-ui/agent-control-bar';
import { Shimmer } from '@/components/ai-elements/shimmer';
import { LiveRagEvidencePanel } from '@/components/app/live-rag-evidence-panel';
import {
  type LiveRagRuntimeState,
  type LiveRagTurn,
  getRuntimeState,
  getSessionTurns,
} from '@/lib/liverag-api';
import { decodeLiveRagDisplayText, getLiveRagDisplayName } from '@/lib/liverag-display';
import { cn } from '@/lib/shadcn/utils';
import { TileLayout } from './tile-view';

const MotionMessage = motion.create(Shimmer);

const BOTTOM_VIEW_MOTION_PROPS: MotionProps = {
  variants: {
    visible: {
      opacity: 1,
      translateY: '0%',
    },
    hidden: {
      opacity: 0,
      translateY: '100%',
    },
  },
  initial: 'hidden',
  animate: 'visible',
  exit: 'hidden',
  transition: {
    duration: 0.3,
    delay: 0.5,
    ease: 'easeOut',
  },
};

const CHAT_MOTION_PROPS: MotionProps = {
  variants: {
    hidden: {
      opacity: 0,
      transition: {
        ease: 'easeOut',
        duration: 0.3,
      },
    },
    visible: {
      opacity: 1,
      transition: {
        delay: 0.2,
        ease: 'easeOut',
        duration: 0.3,
      },
    },
  },
  initial: 'hidden',
  animate: 'visible',
  exit: 'hidden',
};

const SHIMMER_MOTION_PROPS: MotionProps = {
  variants: {
    visible: {
      opacity: 1,
      transition: {
        ease: 'easeIn',
        duration: 0.5,
        delay: 0.8,
      },
    },
    hidden: {
      opacity: 0,
      transition: {
        ease: 'easeIn',
        duration: 0.5,
        delay: 0,
      },
    },
  },
  initial: 'hidden',
  animate: 'visible',
  exit: 'hidden',
};

function getLatestAssistantTurn(turns: LiveRagTurn[]) {
  return [...turns].reverse().find((turn) => turn.assistant_message?.content);
}

function LiveRagFullscreenContext({
  turns,
  runtime,
  hidden,
}: {
  turns: LiveRagTurn[];
  runtime: LiveRagRuntimeState | null;
  hidden: boolean;
}) {
  const latestTurn = getLatestAssistantTurn(turns);
  const latestRag = latestTurn?.rag;
  const latestChunks = latestRag?.evidence_chunks ?? [];
  const userText =
    latestTurn?.user_message?.content ?? runtime?.last_user_text ?? '正在等待语音输入...';
  const assistantText =
    latestTurn?.assistant_message?.content ??
    (runtime?.last_rag_query ? `正在检索知识库：${runtime.last_rag_query}` : '已连接，正在监听');

  return (
    <div
      className={cn(
        'pointer-events-none absolute inset-x-4 bottom-[122px] z-40 flex justify-center transition duration-200 md:bottom-[150px]',
        hidden && 'translate-y-3 opacity-0'
      )}
    >
      <div className="relative grid w-full max-w-2xl gap-1.5 text-center">
        {userText && (
          <div className="flex min-w-0 justify-center gap-2 text-xs md:text-sm">
            <span className="text-muted-foreground font-mono text-[10px] font-bold tracking-wider uppercase">
              You
            </span>
            <span className="text-muted-foreground truncate">{userText}</span>
          </div>
        )}
        {(assistantText || runtime?.last_rag_query) && (
          <div className="flex min-w-0 justify-center gap-2 text-xs font-medium md:text-sm">
            <span className="text-muted-foreground font-mono text-[10px] font-bold tracking-wider uppercase">
              AI
            </span>
            <span className="truncate">
              {assistantText ?? `正在检索知识库：${runtime?.last_rag_query}`}
            </span>
          </div>
        )}
        {latestRag && latestRag.status !== 'not_queried' && (
          <div className="mt-1 grid justify-items-center gap-1.5">
            <div className="text-muted-foreground text-[11px] font-semibold">
              本次回答依据
              <span className="ml-2">
                {latestRag.status === 'hit'
                  ? `${latestRag.evidence_count ?? latestChunks.length} 个来源`
                  : latestRag.status === 'miss'
                    ? '未命中'
                    : '查询失败'}
              </span>
            </div>
            {latestChunks.length > 0 && (
              <div className="flex max-w-full flex-wrap justify-center gap-1.5">
                {latestChunks.slice(0, 3).map((chunk, index) => (
                  <span
                    key={chunk.chunk_id ?? `${chunk.document_id}-${index}`}
                    title={chunk.content_preview}
                    className="bg-background/60 text-muted-foreground max-w-48 truncate rounded-full border px-2 py-1 text-[11px] backdrop-blur-sm"
                  >
                    {chunk.file_path
                      ? getLiveRagDisplayName(chunk.file_path)
                      : decodeLiveRagDisplayText(chunk.document_id) || '知识库片段'}
                    {typeof chunk.score === 'number' ? ` · ${chunk.score.toFixed(2)}` : ''}
                  </span>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export interface AgentSessionView_01Props {
  /**
   * Message shown above the controls before the first chat message is sent.
   *
   * @default '助手正在聆听，可以开始提问'
   */
  preConnectMessage?: string;
  /**
   * Enables or disables the chat toggle and transcript input controls.
   *
   * @default true
   */
  supportsChatInput?: boolean;
  /**
   * Enables or disables camera controls in the bottom control bar.
   *
   * @default true
   */
  supportsVideoInput?: boolean;
  /**
   * Enables or disables screen sharing controls in the bottom control bar.
   *
   * @default true
   */
  supportsScreenShare?: boolean;
  /**
   * Shows a pre-connect buffer state with a shimmer message before messages appear.
   *
   * @default true
   */
  isPreConnectBufferEnabled?: boolean;

  /** Selects the visualizer style rendered in the main tile area. */
  audioVisualizerType?: 'bar' | 'wave' | 'grid' | 'radial' | 'aura';
  /** Primary hex color used by supported audio visualizer variants. */
  audioVisualizerColor?: `#${string}`;
  /** Hue shift intensity used by certain visualizers. */
  audioVisualizerColorShift?: number;
  /** Number of bars to render when `audioVisualizerType` is `bar`. */
  audioVisualizerBarCount?: number;
  /** Number of rows in the visualizer when `audioVisualizerType` is `grid`. */
  audioVisualizerGridRowCount?: number;
  /** Number of columns in the visualizer when `audioVisualizerType` is `grid`. */
  audioVisualizerGridColumnCount?: number;
  /** Number of radial bars when `audioVisualizerType` is `radial`. */
  audioVisualizerRadialBarCount?: number;
  /** Base radius of the radial visualizer when `audioVisualizerType` is `radial`. */
  audioVisualizerRadialRadius?: number;
  /** Stroke width of the wave path when `audioVisualizerType` is `wave`. */
  audioVisualizerWaveLineWidth?: number;
  /** Optional class name merged onto the outer `<section>` container. */
  className?: string;
}

export function AgentSessionView_01({
  preConnectMessage = '助手正在聆听，可以开始提问',
  supportsChatInput = true,
  supportsVideoInput = true,
  supportsScreenShare = true,
  isPreConnectBufferEnabled = true,

  audioVisualizerType,
  audioVisualizerColor,
  audioVisualizerColorShift,
  audioVisualizerBarCount,
  audioVisualizerGridRowCount,
  audioVisualizerGridColumnCount,
  audioVisualizerRadialBarCount,
  audioVisualizerRadialRadius,
  audioVisualizerWaveLineWidth,
  ref,
  className,
  ...props
}: React.ComponentProps<'section'> & AgentSessionView_01Props) {
  const session = useSessionContext();
  const { messages } = useSessionMessages(session);
  const [chatOpen, setChatOpen] = useState(false);
  const [evidenceOpen, setEvidenceOpen] = useState(false);
  const [turns, setTurns] = useState<LiveRagTurn[]>([]);
  const [runtime, setRuntime] = useState<LiveRagRuntimeState | null>(null);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const { state: agentState } = useAgent();
  const livekitMessages = useMemo(() => messages, [messages]);

  useEffect(() => {
    document.body.dataset.liveragSession = 'connected';

    return () => {
      delete document.body.dataset.liveragSession;
    };
  }, []);

  const controls: AgentControlBarControls = {
    leave: true,
    microphone: true,
    chat: supportsChatInput,
    camera: supportsVideoInput,
    screenShare: supportsScreenShare,
  };

  useEffect(() => {
    const lastMessage = messages.at(-1);
    const lastMessageIsLocal = lastMessage?.from?.isLocal === true;

    if (scrollAreaRef.current && lastMessageIsLocal) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight;
    }
  }, [messages]);

  useEffect(() => {
    let cancelled = false;

    const refreshLiveRagState = async () => {
      try {
        const [turnResults, runtimeResult] = await Promise.all([
          getSessionTurns(10),
          getRuntimeState(),
        ]);

        if (cancelled) return;
        setTurns(turnResults);
        setRuntime(runtimeResult);
      } catch {
        if (cancelled) return;
        setRuntime(null);
      }
    };

    void refreshLiveRagState();
    const interval = window.setInterval(refreshLiveRagState, 1800);

    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, []);

  return (
    <section
      ref={ref}
      className={cn(
        'bg-background from-background via-background to-muted/25 fixed inset-0 z-[60] h-svh w-full overflow-hidden bg-linear-to-b',
        className
      )}
      {...props}
    >
      {/* transcript */}

      <div className="absolute top-[150px] bottom-[126px] z-20 flex w-full flex-col md:top-[170px] md:bottom-[142px]">
        <AnimatePresence>
          {chatOpen && (
            <motion.div
              {...CHAT_MOTION_PROPS}
              className="flex h-full w-full flex-col gap-4 space-y-3 transition-opacity duration-300 ease-out"
            >
              <AgentChatTranscript
                agentState={agentState}
                messages={livekitMessages}
                turns={turns}
                className={cn(
                  'mx-auto w-full max-w-[720px]',
                  '[&_.is-user>div]:rounded-[22px] [&>div>div]:px-4 md:[&>div>div]:px-6'
                )}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
      {/* Tile layout */}
      <TileLayout
        chatOpen={chatOpen}
        audioVisualizerType={audioVisualizerType}
        audioVisualizerColor={audioVisualizerColor}
        audioVisualizerColorShift={audioVisualizerColorShift}
        audioVisualizerBarCount={audioVisualizerBarCount}
        audioVisualizerRadialBarCount={audioVisualizerRadialBarCount}
        audioVisualizerRadialRadius={audioVisualizerRadialRadius}
        audioVisualizerGridRowCount={audioVisualizerGridRowCount}
        audioVisualizerGridColumnCount={audioVisualizerGridColumnCount}
        audioVisualizerWaveLineWidth={audioVisualizerWaveLineWidth}
      />
      <LiveRagFullscreenContext turns={turns} runtime={runtime} hidden={chatOpen} />
      {/* Bottom */}
      <motion.div
        {...BOTTOM_VIEW_MOTION_PROPS}
        className="absolute inset-x-3 bottom-5 z-50 md:inset-x-12 md:bottom-6"
      >
        {/* Pre-connect message */}
        {isPreConnectBufferEnabled && (
          <AnimatePresence>
            {messages.length === 0 && (
              <MotionMessage
                key="pre-connect-message"
                duration={2}
                aria-hidden={messages.length > 0}
                {...SHIMMER_MOTION_PROPS}
                className="text-muted-foreground pointer-events-none mx-auto block w-full max-w-[720px] pb-3 text-center text-xs font-semibold"
              >
                {preConnectMessage}
              </MotionMessage>
            )}
          </AnimatePresence>
        )}
        <div className="relative mx-auto max-w-[720px]">
          <AgentControlBar
            variant="livekit"
            controls={controls}
            isChatOpen={chatOpen}
            isConnected={session.isConnected}
            onDisconnect={session.end}
            onIsChatOpenChange={setChatOpen}
            leadingControl={
              <button
                type="button"
                aria-label="打开回答依据"
                title="回答依据"
                onClick={() => setEvidenceOpen(true)}
                className="border-input bg-muted text-foreground hover:bg-foreground/10 focus-visible:ring-ring inline-flex size-10 shrink-0 items-center justify-center rounded-full border transition-colors focus-visible:ring-2 focus-visible:outline-none"
              >
                <DatabaseIcon className="size-4" />
              </button>
            }
            className="bg-background/95 shadow-[0_14px_48px_rgba(0,0,0,0.08)] backdrop-blur-xl"
          />
        </div>
      </motion.div>
      <LiveRagEvidencePanel
        open={evidenceOpen}
        onOpenChange={setEvidenceOpen}
        turns={turns}
        runtime={runtime}
      />
    </section>
  );
}
