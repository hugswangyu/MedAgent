'use client';

import {
  type FormEvent,
  type KeyboardEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { TokenSource } from 'livekit-client';
import {
  BookOpenTextIcon,
  ChevronDownIcon,
  FileHeartIcon,
  MicIcon,
  MicOffIcon,
  SendIcon,
  ShieldCheckIcon,
  SquareIcon,
  StethoscopeIcon,
  Volume2Icon,
} from 'lucide-react';
import {
  useAgent,
  useSession,
  useSessionContext,
  useSessionMessages,
} from '@livekit/components-react';
import type { AppConfig } from '@/app-config';
import { AgentSessionProvider } from '@/components/agents-ui/agent-session-provider';
import { StartAudioButton } from '@/components/agents-ui/start-audio-button';
import { MessageResponse } from '@/components/ai-elements/message';
import { LiveRagKnowledgePanel } from '@/components/app/live-rag-knowledge-panel';
import { useInputControls } from '@/hooks/agents-ui/use-agent-control-bar';
import {
  type LiveRagTurn,
  getCurrentVoiceSessionEvidence,
  getSessionKnowledgeBase,
} from '@/lib/liverag-api';
import { type ChatEvent, streamTextChat } from '@/lib/medagent-api';
import {
  type ConsultationEvidence,
  type ConsultationInputMode,
  type ConsultationMessage,
  type ConversationIdentity,
  VOICE_STATE_LABELS,
  mergeConversationTimeline,
  resolveVoiceConnectionState,
} from '@/lib/medical-conversation';

const SUGGESTIONS = [
  '最近总是头晕，需要注意什么？',
  '帮我整理这份检查结果',
  '根据我的资料，复诊时该问什么？',
];

function asIsoDate(value: unknown, fallback: string) {
  if (typeof value !== 'string' && typeof value !== 'number') return fallback;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? fallback : date.toISOString();
}

function voiceTurnMessages(turns: LiveRagTurn[]): ConsultationMessage[] {
  return turns.flatMap((turn, index) => {
    const fallback = new Date(Date.now() - (turns.length - index) * 10).toISOString();
    const evidence: ConsultationEvidence[] = [
      ...(turn.rag?.evidence_chunks ?? []).map((chunk, chunkIndex) => ({
        id: chunk.chunk_id ?? `${turn.turn_index}-chunk-${chunkIndex}`,
        title: chunk.file_path?.split(/[\\/]/).at(-1) || '医学资料片段',
        preview: chunk.content_preview,
        score: chunk.score,
      })),
      ...(turn.rag?.evidence_documents ?? []).map((document, documentIndex) => ({
        id: document.document_id ?? `${turn.turn_index}-document-${documentIndex}`,
        title: document.title || document.file_path?.split(/[\\/]/).at(-1) || '医学资料',
      })),
    ];
    const items: ConsultationMessage[] = [];
    if (turn.user_message?.content) {
      items.push({
        id: `voice-turn-${turn.turn_index}-user`,
        role: 'user',
        channel: 'voice',
        content: turn.user_message.content,
        createdAt: asIsoDate(turn.user_message.timestamp, fallback),
      });
    }
    if (turn.assistant_message?.content) {
      items.push({
        id: `voice-turn-${turn.turn_index}-assistant`,
        role: 'assistant',
        channel: 'voice',
        content: turn.assistant_message.content,
        createdAt: asIsoDate(turn.assistant_message.timestamp, fallback),
        evidence,
      });
    }
    return items;
  });
}

function chatEventEvidence(event: ChatEvent, index: number): ConsultationEvidence[] {
  const rawSources = Array.isArray(event.sources)
    ? event.sources
    : Array.isArray(event.evidence)
      ? event.evidence
      : [];
  const sources = rawSources.filter(
    (source): source is Record<string, unknown> => typeof source === 'object' && source !== null
  );
  if (sources.length === 0) {
    return [
      {
        id: `text-evidence-${index}`,
        title: '医学资料检索记录',
        preview: typeof event.content === 'string' ? event.content : undefined,
      },
    ];
  }
  return sources.map((source, sourceIndex) => ({
    id: String(source.evidence_id ?? source.document_id ?? `${index}-${sourceIndex}`),
    title: String(source.title ?? source.filename ?? source.source_name ?? '医学资料'),
    preview:
      typeof source.content_preview === 'string'
        ? source.content_preview
        : typeof source.content === 'string'
          ? source.content
          : undefined,
    score: typeof source.score === 'number' ? source.score : undefined,
  }));
}

function AnswerEvidence({ evidence }: { evidence: ConsultationEvidence[] }) {
  if (evidence.length === 0) return null;
  return (
    <details className="medical-evidence mt-3 border-t border-slate-200/80 pt-3">
      <summary className="flex cursor-pointer list-none items-center gap-2 rounded-md text-xs font-semibold text-slate-600 focus-visible:ring-2 focus-visible:ring-[#587a8e] focus-visible:outline-none">
        <BookOpenTextIcon className="size-3.5" aria-hidden="true" />
        回答依据 · {evidence.length} 项
        <ChevronDownIcon className="ml-auto size-3.5 transition-transform" aria-hidden="true" />
      </summary>
      <div className="mt-3 grid gap-2">
        {evidence.map((item) => (
          <div key={item.id} className="rounded-xl bg-slate-50 px-3 py-2">
            <div className="flex items-center justify-between gap-3 text-xs font-medium text-slate-700">
              <span className="truncate">{item.title}</span>
              {typeof item.score === 'number' && (
                <span className="shrink-0 text-[11px] text-slate-400">
                  相关度 {item.score.toFixed(2)}
                </span>
              )}
            </div>
            {item.preview && (
              <p className="mt-1 line-clamp-3 text-xs leading-5 text-slate-500">{item.preview}</p>
            )}
          </div>
        ))}
      </div>
    </details>
  );
}

function ConsultationInner({
  identity,
  active,
}: {
  identity: ConversationIdentity;
  active: boolean;
}) {
  const session = useSessionContext();
  const { messages: liveMessages } = useSessionMessages(session);
  const { state: agentState } = useAgent();
  const { microphoneToggle } = useInputControls();
  const [inputMode, setInputMode] = useState<ConsultationInputMode>('text');
  const [input, setInput] = useState('');
  const [textMessages, setTextMessages] = useState<ConsultationMessage[]>([]);
  const [voiceMessages, setVoiceMessages] = useState<ConsultationMessage[]>([]);
  const [knowledgeOpen, setKnowledgeOpen] = useState(false);
  const [knowledgeName, setKnowledgeName] = useState('正在读取医学资料');
  const [pending, setPending] = useState(false);
  const [voiceStarting, setVoiceStarting] = useState(false);
  const [error, setError] = useState('');
  const abortRef = useRef<AbortController | null>(null);
  const timelineEndRef = useRef<HTMLDivElement>(null);

  const refreshKnowledge = useCallback(async () => {
    try {
      const state = await getSessionKnowledgeBase();
      setKnowledgeName(state.active_session?.name ?? state.configured?.name ?? '尚未选择医学资料');
    } catch {
      setKnowledgeName('尚未选择医学资料');
    }
  }, []);

  const refreshVoiceTurns = useCallback(async () => {
    try {
      const turns = await getCurrentVoiceSessionEvidence(80);
      setVoiceMessages((current) => mergeConversationTimeline(current, voiceTurnMessages(turns)));
    } catch {
      // No voice child session exists yet; text consultation remains available.
    }
  }, []);

  useEffect(() => {
    void refreshKnowledge();
  }, [refreshKnowledge]);

  useEffect(() => {
    if (!session.isConnected) return;
    void refreshVoiceTurns();
    const interval = window.setInterval(refreshVoiceTurns, 1400);
    return () => window.clearInterval(interval);
  }, [refreshVoiceTurns, session.isConnected]);

  useEffect(() => {
    const next = liveMessages
      .filter((message) => Boolean(message.message?.trim()))
      .map<ConsultationMessage>((message, index) => ({
        id: `voice-live-${message.id ?? index}`,
        role: message.from?.isLocal ? 'user' : 'assistant',
        channel: 'voice',
        content: message.message,
        createdAt: asIsoDate(message.timestamp, new Date().toISOString()),
      }));
    setVoiceMessages((current) => mergeConversationTimeline(current, next));
  }, [liveMessages]);

  const timeline = useMemo(
    () => mergeConversationTimeline(textMessages, voiceMessages),
    [textMessages, voiceMessages]
  );

  useEffect(() => {
    timelineEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [timeline]);

  const voiceState = resolveVoiceConnectionState(agentState, session.isConnected, voiceStarting);

  async function startOrToggleVoice() {
    setInputMode('voice');
    setError('');
    if (!session.isConnected) {
      setVoiceStarting(true);
      try {
        await session.start();
      } catch (reason) {
        setError(reason instanceof Error ? reason.message : '暂时无法连接语音');
        setKnowledgeOpen(true);
      } finally {
        setVoiceStarting(false);
      }
      return;
    }
    try {
      await microphoneToggle.toggle(!microphoneToggle.enabled);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '无法切换麦克风');
    }
  }

  async function stopVoice() {
    await refreshVoiceTurns();
    session.end();
    setInputMode('text');
  }

  async function send(event?: FormEvent) {
    event?.preventDefault();
    const message = input.trim();
    if (!message || pending) return;
    const now = new Date().toISOString();
    const assistantId = `text-assistant-${crypto.randomUUID()}`;
    setInput('');
    setInputMode('text');
    setError('');
    setPending(true);
    setTextMessages((items) => [
      ...items,
      {
        id: `text-user-${crypto.randomUUID()}`,
        role: 'user',
        channel: 'text',
        content: message,
        createdAt: now,
      },
      {
        id: assistantId,
        role: 'assistant',
        channel: 'text',
        content: '',
        createdAt: new Date(Date.now() + 1).toISOString(),
        pending: true,
      },
    ]);
    const controller = new AbortController();
    abortRef.current = controller;
    let evidenceIndex = 0;
    try {
      await streamTextChat(
        { message, session_id: identity.textSessionId },
        (item) => {
          if (item.type === 'content' && typeof item.content === 'string') {
            setTextMessages((items) =>
              items.map((entry) =>
                entry.id === assistantId
                  ? { ...entry, content: entry.content + item.content }
                  : entry
              )
            );
          }
          if (
            item.type?.includes('evidence') ||
            item.type?.includes('trace') ||
            'evidence' in item ||
            'sources' in item
          ) {
            const nextEvidence = chatEventEvidence(item, evidenceIndex++);
            setTextMessages((items) =>
              items.map((entry) =>
                entry.id === assistantId
                  ? { ...entry, evidence: [...(entry.evidence ?? []), ...nextEvidence] }
                  : entry
              )
            );
          }
          if (item.type === 'error') {
            setError(String(item.content ?? '本次咨询暂时失败'));
          }
        },
        controller.signal
      );
    } catch (reason) {
      if (!(reason instanceof DOMException && reason.name === 'AbortError')) {
        setError(reason instanceof Error ? reason.message : '本次咨询暂时失败');
      }
    } finally {
      setTextMessages((items) =>
        items.map((entry) => (entry.id === assistantId ? { ...entry, pending: false } : entry))
      );
      abortRef.current = null;
      setPending(false);
    }
  }

  function stopGeneration() {
    abortRef.current?.abort();
    setPending(false);
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      void send();
    }
  }

  return (
    <section className={active ? 'flex min-h-0 flex-1 flex-col' : 'hidden'} aria-label="医疗咨询">
      <header className="flex min-h-16 flex-wrap items-center gap-3 border-b border-slate-200/80 bg-white/85 px-4 py-3 backdrop-blur md:px-7">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <h1 className="text-base font-semibold text-slate-800 md:text-lg">医疗咨询</h1>
            <span className="rounded-full bg-emerald-50 px-2 py-1 text-[11px] font-medium text-emerald-700">
              隐私保护
            </span>
          </div>
          <p className="mt-0.5 truncate text-xs text-slate-500">
            咨询编号 {identity.conversationId.slice(-8)}
          </p>
        </div>
        <button
          type="button"
          onClick={() => setKnowledgeOpen(true)}
          aria-label="选择病历或医学资料"
          className="ml-auto inline-flex max-w-[52vw] items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600 shadow-sm transition hover:border-slate-300 hover:bg-slate-50 focus-visible:ring-2 focus-visible:ring-[#587a8e] focus-visible:outline-none md:max-w-xs"
        >
          <FileHeartIcon className="size-4 shrink-0 text-[#587a8e]" aria-hidden="true" />
          <span className="truncate">{knowledgeName}</span>
        </button>
      </header>

      <div className="medical-chat-scroll flex-1 overflow-y-auto px-3 py-5 md:px-8 md:py-7">
        <div className="mx-auto flex w-full max-w-3xl flex-col gap-5">
          {timeline.length === 0 && (
            <div className="mx-auto flex max-w-xl flex-col items-center px-3 py-8 text-center md:py-14">
              <div className="grid size-14 place-items-center rounded-2xl bg-[#eaf1f4] text-[#46697d]">
                <StethoscopeIcon className="size-7" aria-hidden="true" />
              </div>
              <h2 className="mt-5 text-xl font-semibold text-slate-800">今天有什么可以帮你？</h2>
              <p className="mt-2 max-w-md text-sm leading-6 text-slate-500">
                可以描述身体不适、解读检查结果，或结合病历资料整理复诊问题。
              </p>
              <div className="mt-6 grid w-full gap-2 sm:grid-cols-3">
                {SUGGESTIONS.map((suggestion) => (
                  <button
                    key={suggestion}
                    type="button"
                    onClick={() => setInput(suggestion)}
                    className="rounded-xl border border-slate-200 bg-white px-3 py-3 text-left text-xs leading-5 text-slate-600 shadow-sm transition hover:border-[#9bb0bd] hover:bg-[#f7fafb] focus-visible:ring-2 focus-visible:ring-[#587a8e] focus-visible:outline-none"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          )}

          {timeline.map((message) => (
            <article
              key={message.id}
              className={`flex gap-2.5 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              {message.role === 'assistant' && (
                <div className="mt-1 grid size-8 shrink-0 place-items-center rounded-xl bg-[#dfe9ed] text-[#496b7e]">
                  <StethoscopeIcon className="size-4" aria-hidden="true" />
                </div>
              )}
              <div
                className={`max-w-[86%] rounded-2xl px-4 py-3 text-sm leading-7 shadow-sm md:max-w-[78%] md:px-5 ${
                  message.role === 'user'
                    ? 'rounded-br-md bg-[#5c7f92] text-white'
                    : 'rounded-bl-md border border-slate-200/80 bg-white text-slate-700'
                }`}
              >
                <div className="mb-1.5 flex items-center gap-2 text-[10px] font-medium tracking-wide opacity-70">
                  {message.channel === 'voice' ? (
                    <>
                      <Volume2Icon className="size-3" aria-hidden="true" />
                      语音转写
                    </>
                  ) : message.role === 'user' ? (
                    '你'
                  ) : (
                    'MedAgent 医疗助手'
                  )}
                </div>
                {message.content ? (
                  message.role === 'assistant' ? (
                    <MessageResponse>{message.content}</MessageResponse>
                  ) : (
                    <p className="whitespace-pre-wrap">{message.content}</p>
                  )
                ) : (
                  <p role="status" className="animate-pulse">
                    正在思考，请稍候…
                  </p>
                )}
                {message.role === 'assistant' && (
                  <AnswerEvidence evidence={message.evidence ?? []} />
                )}
              </div>
            </article>
          ))}
          <div ref={timelineEndRef} />
        </div>
      </div>

      <div className="border-t border-slate-200/80 bg-[#fbfcfc]/95 px-3 pt-3 pb-[max(12px,env(safe-area-inset-bottom))] backdrop-blur md:px-8 md:pb-5">
        <div className="mx-auto max-w-3xl">
          <div
            className="mb-2 flex min-h-6 items-center gap-2 text-xs text-slate-500"
            aria-live="polite"
          >
            <span
              className={`size-2 rounded-full ${
                voiceState === 'disconnected' ? 'bg-slate-300' : 'bg-emerald-500'
              }`}
              aria-hidden="true"
            />
            <span>{VOICE_STATE_LABELS[voiceState]}</span>
            {session.isConnected && !microphoneToggle.enabled && (
              <span className="font-medium text-amber-700">· 麦克风已静音</span>
            )}
            {session.isConnected && (
              <button
                type="button"
                onClick={() => void stopVoice()}
                className="ml-auto rounded-md px-2 py-1 font-medium text-rose-600 hover:bg-rose-50 focus-visible:ring-2 focus-visible:ring-[#587a8e] focus-visible:outline-none"
                aria-label="结束语音咨询"
              >
                结束语音
              </button>
            )}
          </div>

          <form
            onSubmit={send}
            className="flex items-end gap-2 rounded-2xl border border-slate-200 bg-white p-2 shadow-[0_8px_30px_rgba(55,72,82,0.08)] transition focus-within:border-[#7594a5] focus-within:ring-4 focus-within:ring-[#7594a5]/15"
          >
            <button
              type="button"
              onClick={() => void startOrToggleVoice()}
              aria-label={
                session.isConnected
                  ? microphoneToggle.enabled
                    ? '关闭麦克风'
                    : '打开麦克风'
                  : '开始语音咨询'
              }
              aria-pressed={session.isConnected && microphoneToggle.enabled}
              className={`grid size-10 shrink-0 place-items-center rounded-xl transition focus-visible:ring-2 focus-visible:ring-[#587a8e] focus-visible:outline-none ${
                inputMode === 'voice'
                  ? 'bg-[#dfeaed] text-[#3d657b]'
                  : 'text-slate-500 hover:bg-slate-100'
              }`}
            >
              {session.isConnected && microphoneToggle.enabled ? (
                <MicIcon className="size-5" aria-hidden="true" />
              ) : (
                <MicOffIcon className="size-5" aria-hidden="true" />
              )}
            </button>
            <textarea
              value={input}
              onFocus={() => setInputMode('text')}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={handleKeyDown}
              rows={1}
              aria-label="输入医疗咨询内容"
              placeholder="描述症状、检查结果或想了解的健康问题…"
              className="max-h-32 min-h-10 flex-1 resize-none bg-transparent px-1 py-2 text-sm leading-6 text-slate-700 outline-none placeholder:text-slate-400"
            />
            {pending ? (
              <button
                type="button"
                onClick={stopGeneration}
                aria-label="停止生成回答"
                className="grid size-10 shrink-0 place-items-center rounded-xl bg-slate-700 text-white transition hover:bg-slate-800 focus-visible:ring-2 focus-visible:ring-[#587a8e] focus-visible:ring-offset-2 focus-visible:outline-none"
              >
                <SquareIcon className="size-4 fill-current" aria-hidden="true" />
              </button>
            ) : (
              <button
                type="submit"
                disabled={!input.trim()}
                aria-label="发送咨询"
                className="grid size-10 shrink-0 place-items-center rounded-xl bg-[#587a8e] text-white transition hover:bg-[#46697d] focus-visible:ring-2 focus-visible:ring-[#587a8e] focus-visible:ring-offset-2 focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-35"
              >
                <SendIcon className="size-4" aria-hidden="true" />
              </button>
            )}
          </form>
          {error && (
            <p role="alert" className="mt-2 text-xs text-rose-600">
              {error}
            </p>
          )}
          <p className="mt-2 flex items-start justify-center gap-1.5 text-center text-[10px] leading-4 text-slate-400">
            <ShieldCheckIcon className="mt-0.5 size-3 shrink-0" aria-hidden="true" />
            内容仅供健康参考，不能替代医生面诊；如有胸痛、呼吸困难等紧急情况，请立即联系急救。
          </p>
        </div>
      </div>
      <LiveRagKnowledgePanel
        open={knowledgeOpen}
        onOpenChange={(open) => {
          setKnowledgeOpen(open);
          if (!open) void refreshKnowledge();
        }}
      />
    </section>
  );
}

export function MedicalConsultation({
  appConfig,
  identity,
  active,
}: {
  appConfig: AppConfig;
  identity: ConversationIdentity;
  active: boolean;
}) {
  const tokenSource = useMemo(
    () => TokenSource.endpoint(`/api/token?conversation_id=${identity.voiceClientId}`),
    [identity.voiceClientId]
  );
  const session = useSession(
    tokenSource,
    appConfig.agentName
      ? {
          agentName: appConfig.agentName,
          agentConnectTimeoutMilliseconds: 10 * 60 * 1000,
        }
      : undefined
  );

  return (
    <AgentSessionProvider session={session}>
      <ConsultationInner identity={identity} active={active} />
      <StartAudioButton label="点击开启语音播报" />
    </AgentSessionProvider>
  );
}
