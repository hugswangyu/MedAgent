'use client';

import { type FormEvent, useCallback, useEffect, useMemo, useState } from 'react';
import type { AppConfig } from '@/app-config';
import { App as VoiceApp } from '@/components/app/app';
import { LiveRagEvidencePanel } from '@/components/app/live-rag-evidence-panel';
import { LiveRagKnowledgePanel } from '@/components/app/live-rag-knowledge-panel';
import {
  type LiveRagRuntimeState,
  type LiveRagTurn,
  getCurrentVoiceSessionEvidence,
  getRuntimeState,
} from '@/lib/liverag-api';
import {
  type ChatEvent,
  type MedAgentUser,
  type MedicalMemory,
  correctMemory,
  deleteMemory,
  getCurrentUser,
  listMemories,
  login,
  logout,
  streamTextChat,
  transitionMemory,
} from '@/lib/medagent-api';

type Tab = 'text' | 'voice' | 'knowledge' | 'evidence' | 'memory';

const tabs: Array<{ id: Tab; label: string }> = [
  { id: 'text', label: '文字问诊' },
  { id: 'voice', label: '实时语音' },
  { id: 'knowledge', label: '个人知识库' },
  { id: 'evidence', label: '证据' },
  { id: 'memory', label: '记忆' },
];

function LoginView({ onAuthenticated }: { onAuthenticated: (user: MedAgentUser) => void }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [pending, setPending] = useState(false);
  const [error, setError] = useState('');

  async function submit(event: FormEvent) {
    event.preventDefault();
    setPending(true);
    setError('');
    try {
      await login(username.trim(), password);
      onAuthenticated(await getCurrentUser());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '登录失败');
    } finally {
      setPending(false);
    }
  }

  return (
    <main className="bg-background grid min-h-svh place-items-center px-5">
      <form
        onSubmit={submit}
        className="border-border bg-card w-full max-w-sm rounded-3xl border p-8 shadow-sm"
      >
        <p className="text-muted-foreground text-xs font-semibold tracking-[0.22em] uppercase">
          MedAgent × MedLive
        </p>
        <h1 className="mt-3 text-2xl font-semibold">登录医疗助手</h1>
        <p className="text-muted-foreground mt-2 text-sm">
          同一身份用于文字、语音、个人知识库、证据与受控记忆。
        </p>
        <label className="mt-7 block text-sm font-medium">
          用户名
          <input
            required
            autoComplete="username"
            value={username}
            onChange={(event) => setUsername(event.target.value)}
            className="border-input bg-background mt-2 w-full rounded-xl border px-3 py-2.5 outline-none focus:ring-2"
          />
        </label>
        <label className="mt-4 block text-sm font-medium">
          密码
          <input
            required
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            className="border-input bg-background mt-2 w-full rounded-xl border px-3 py-2.5 outline-none focus:ring-2"
          />
        </label>
        {error && <p className="mt-4 text-sm text-red-600">{error}</p>}
        <button
          disabled={pending}
          className="bg-foreground text-background mt-6 w-full rounded-xl px-4 py-2.5 text-sm font-semibold disabled:opacity-50"
        >
          {pending ? '正在登录…' : '登录'}
        </button>
        <a
          className="text-muted-foreground mt-5 block text-center text-xs underline"
          href="/legacy"
        >
          打开旧版 Vue 界面
        </a>
      </form>
    </main>
  );
}

interface TextMessage {
  role: 'user' | 'assistant';
  content: string;
}

function TextChat() {
  const [sessionId] = useState(() => 'web_' + crypto.randomUUID());
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<TextMessage[]>([]);
  const [evidence, setEvidence] = useState<ChatEvent[]>([]);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState('');

  async function send(event: FormEvent) {
    event.preventDefault();
    const message = input.trim();
    if (!message || pending) return;
    setInput('');
    setError('');
    setPending(true);
    setMessages((items) => [
      ...items,
      { role: 'user', content: message },
      { role: 'assistant', content: '' },
    ]);
    try {
      await streamTextChat({ message, session_id: sessionId }, (item) => {
        if (item.type === 'content' && typeof item.content === 'string') {
          setMessages((items) => {
            const next = [...items];
            const last = next[next.length - 1];
            next[next.length - 1] = { ...last, content: last.content + item.content };
            return next;
          });
        }
        if (
          item.type?.includes('evidence') ||
          item.type?.includes('trace') ||
          'evidence' in item ||
          'sources' in item
        ) {
          setEvidence((items) => [...items, item]);
        }
        if (item.type === 'error') {
          setError(String(item.content ?? '文字问诊失败'));
        }
      });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '文字问诊失败');
    } finally {
      setPending(false);
    }
  }

  return (
    <section className="mx-auto flex min-h-[calc(100svh-5rem)] max-w-5xl flex-col gap-4 px-4 py-6">
      <div className="border-border bg-card flex-1 space-y-4 overflow-auto rounded-3xl border p-5">
        {messages.length === 0 && (
          <div className="text-muted-foreground grid min-h-72 place-items-center text-center text-sm">
            <div>
              <p className="text-foreground text-lg font-semibold">文字医疗问答</p>
              <p className="mt-2">保留 MedAgent ReAct，并共享登录、会话和受控记忆。</p>
            </div>
          </div>
        )}
        {messages.map((message, index) => (
          <div
            key={index}
            className={
              'max-w-[85%] rounded-2xl px-4 py-3 text-sm whitespace-pre-wrap ' +
              (message.role === 'user'
                ? 'bg-foreground text-background ml-auto'
                : 'bg-muted text-foreground')
            }
          >
            {message.content || (pending ? '思考中…' : '')}
          </div>
        ))}
        {evidence.length > 0 && (
          <details className="border-border rounded-2xl border p-4 text-xs">
            <summary className="cursor-pointer font-semibold">
              本会话证据与检索轨迹（{evidence.length}）
            </summary>
            <pre className="mt-3 max-h-64 overflow-auto whitespace-pre-wrap">
              {JSON.stringify(evidence, null, 2)}
            </pre>
          </details>
        )}
      </div>
      <form onSubmit={send} className="border-border bg-card flex gap-3 rounded-2xl border p-3">
        <textarea
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder="描述问题；当前胸痛、呼吸困难等紧急症状请立即联系急救…"
          rows={2}
          className="flex-1 resize-none bg-transparent px-2 py-1 text-sm outline-none"
        />
        <button
          disabled={pending || !input.trim()}
          className="bg-foreground text-background rounded-xl px-5 text-sm font-semibold disabled:opacity-40"
        >
          发送
        </button>
      </form>
      {error && <p className="text-sm text-red-600">{error}</p>}
    </section>
  );
}

function KnowledgeView() {
  const [open, setOpen] = useState(true);
  return (
    <section className="grid min-h-[calc(100svh-5rem)] place-items-center p-6 text-center">
      <div>
        <h2 className="text-xl font-semibold">个人知识库</h2>
        <p className="text-muted-foreground mt-2 text-sm">
          创建、选择知识库并管理文档；语音会话使用当前选中的知识库。
        </p>
        <button
          onClick={() => setOpen(true)}
          className="bg-foreground text-background mt-5 rounded-xl px-5 py-2.5 text-sm font-semibold"
        >
          打开知识库管理
        </button>
      </div>
      <LiveRagKnowledgePanel open={open} onOpenChange={setOpen} />
    </section>
  );
}

function EvidenceView() {
  const [turns, setTurns] = useState<LiveRagTurn[]>([]);
  const [runtime, setRuntime] = useState<LiveRagRuntimeState | null>(null);
  const [open, setOpen] = useState(false);
  const [error, setError] = useState('');

  const refresh = useCallback(async () => {
    setError('');
    try {
      const [nextTurns, nextRuntime] = await Promise.all([
        getCurrentVoiceSessionEvidence(50),
        getRuntimeState(),
      ]);
      setTurns(nextTurns);
      setRuntime(nextRuntime);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '证据读取失败');
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const evidenceTurns = useMemo(
    () => turns.filter((turn) => (turn.rag?.evidence_count ?? 0) > 0),
    [turns]
  );

  return (
    <section className="mx-auto max-w-4xl p-6">
      <div className="border-border bg-card rounded-3xl border p-6">
        <h2 className="text-xl font-semibold">证据管理</h2>
        <p className="text-muted-foreground mt-2 text-sm">
          最近 {turns.length} 个语音轮次中有 {evidenceTurns.length} 个包含证据。
        </p>
        <div className="mt-5 flex gap-3">
          <button
            onClick={() => void refresh()}
            className="border-border rounded-xl border px-4 py-2 text-sm"
          >
            刷新
          </button>
          <button
            onClick={() => setOpen(true)}
            className="bg-foreground text-background rounded-xl px-4 py-2 text-sm font-semibold"
          >
            查看证据详情
          </button>
        </div>
        {error && <p className="mt-4 text-sm text-red-600">{error}</p>}
      </div>
      <LiveRagEvidencePanel open={open} onOpenChange={setOpen} turns={turns} runtime={runtime} />
    </section>
  );
}

function MemoryView() {
  const [items, setItems] = useState<MedicalMemory[]>([]);
  const [error, setError] = useState('');
  const [pendingId, setPendingId] = useState('');

  const refresh = useCallback(async () => {
    try {
      setError('');
      setItems((await listMemories()).memories);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '记忆读取失败');
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  async function act(memory: MedicalMemory, action: 'confirm' | 'reject' | 'correct' | 'delete') {
    setPendingId(memory.memory_id);
    setError('');
    try {
      if (action === 'confirm' || action === 'reject') {
        await transitionMemory(memory.memory_id, action);
      } else if (action === 'correct') {
        const content = window.prompt('输入纠正后的事实', memory.content)?.trim();
        if (!content) return;
        await correctMemory(memory.memory_id, content);
      } else if (window.confirm('确认删除这条记忆？')) {
        await deleteMemory(memory.memory_id);
      } else {
        return;
      }
      await refresh();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '记忆操作失败');
    } finally {
      setPendingId('');
    }
  }

  return (
    <section className="mx-auto max-w-5xl p-6">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-semibold">受控医疗记忆</h2>
          <p className="text-muted-foreground mt-1 text-sm">
            确认、拒绝、纠正、删除或导出你的医疗事实候选。
          </p>
        </div>
        <button
          type="button"
          onClick={() => window.location.assign('/api/medagent/memories/export')}
          className="border-border rounded-xl border px-4 py-2 text-sm"
        >
          导出
        </button>
      </div>
      <div className="mt-5 space-y-3">
        {items.map((memory) => (
          <article key={memory.memory_id} className="border-border bg-card rounded-2xl border p-4">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <span className="bg-muted rounded-full px-2 py-1 text-xs">{memory.status}</span>
                <span className="text-muted-foreground ml-2 text-xs">{memory.memory_type}</span>
                <p className="mt-3 text-sm">{memory.content}</p>
              </div>
              <div className="flex flex-wrap gap-2">
                {memory.status === 'proposed' && (
                  <>
                    <button
                      disabled={pendingId === memory.memory_id}
                      onClick={() => void act(memory, 'confirm')}
                      className="rounded-lg border px-3 py-1.5 text-xs"
                    >
                      确认
                    </button>
                    <button
                      disabled={pendingId === memory.memory_id}
                      onClick={() => void act(memory, 'reject')}
                      className="rounded-lg border px-3 py-1.5 text-xs"
                    >
                      拒绝
                    </button>
                  </>
                )}
                <button
                  disabled={pendingId === memory.memory_id}
                  onClick={() => void act(memory, 'correct')}
                  className="rounded-lg border px-3 py-1.5 text-xs"
                >
                  纠正
                </button>
                <button
                  disabled={pendingId === memory.memory_id}
                  onClick={() => void act(memory, 'delete')}
                  className="rounded-lg border px-3 py-1.5 text-xs text-red-600"
                >
                  删除
                </button>
              </div>
            </div>
          </article>
        ))}
        {items.length === 0 && !error && (
          <p className="text-muted-foreground rounded-2xl border border-dashed p-8 text-center text-sm">
            暂无受控记忆。
          </p>
        )}
        {error && <p className="text-sm text-red-600">{error}</p>}
      </div>
    </section>
  );
}

export function UnifiedApp({ appConfig }: { appConfig: AppConfig }) {
  const [user, setUser] = useState<MedAgentUser | null>(null);
  const [checking, setChecking] = useState(true);
  const [tab, setTab] = useState<Tab>('text');

  useEffect(() => {
    getCurrentUser()
      .then(setUser)
      .catch(() => setUser(null))
      .finally(() => setChecking(false));
  }, []);

  if (checking) {
    return <main className="grid min-h-svh place-items-center text-sm">正在检查登录状态…</main>;
  }
  if (!user) return <LoginView onAuthenticated={setUser} />;

  return (
    <div className="bg-background min-h-svh">
      <header className="border-border bg-background/95 sticky top-0 z-[100] flex h-16 items-center gap-3 border-b px-4 backdrop-blur">
        <div className="mr-auto">
          <p className="text-sm font-semibold">MedAgent</p>
          <p className="text-muted-foreground text-[11px]">{user.username}</p>
        </div>
        <nav className="hidden gap-1 md:flex">
          {tabs.map((item) => (
            <button
              key={item.id}
              onClick={() => setTab(item.id)}
              className={
                'rounded-lg px-3 py-2 text-sm ' +
                (tab === item.id ? 'bg-foreground text-background' : 'hover:bg-muted')
              }
            >
              {item.label}
            </button>
          ))}
        </nav>
        <select
          value={tab}
          onChange={(event) => setTab(event.target.value as Tab)}
          className="border-border bg-background rounded-lg border p-2 text-sm md:hidden"
        >
          {tabs.map((item) => (
            <option key={item.id} value={item.id}>
              {item.label}
            </option>
          ))}
        </select>
        <a href="/legacy" className="text-muted-foreground text-xs underline">
          旧版
        </a>
        <button
          onClick={async () => {
            await logout();
            setUser(null);
          }}
          className="text-xs"
        >
          退出
        </button>
      </header>
      {tab === 'text' && <TextChat />}
      {tab === 'voice' && (
        <div className="relative min-h-[calc(100svh-4rem)]">
          <VoiceApp appConfig={appConfig} />
        </div>
      )}
      {tab === 'knowledge' && <KnowledgeView />}
      {tab === 'evidence' && <EvidenceView />}
      {tab === 'memory' && <MemoryView />}
    </div>
  );
}
