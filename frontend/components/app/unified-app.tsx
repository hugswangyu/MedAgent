'use client';

import { type FormEvent, useCallback, useEffect, useMemo, useState } from 'react';
import {
  BookOpenTextIcon,
  Clock3Icon,
  FileHeartIcon,
  HeartPulseIcon,
  LogOutIcon,
  MenuIcon,
  SettingsIcon,
  ShieldCheckIcon,
  StethoscopeIcon,
  UserRoundIcon,
  XIcon,
} from 'lucide-react';
import type { AppConfig } from '@/app-config';
import { LiveRagKnowledgePanel } from '@/components/app/live-rag-knowledge-panel';
import { LiveRagSettingsPanel } from '@/components/app/live-rag-settings-panel';
import { MedAgentLogo } from '@/components/app/medagent-logo';
import { MedicalConsultation } from '@/components/app/medical-consultation';
import {
  type MedAgentUser,
  type MedicalMemory,
  type SessionDetail,
  type SessionSummary,
  correctMemory,
  deleteMemory,
  getCurrentUser,
  getSession,
  listMemories,
  listSessions,
  login,
  logout,
  register,
  transitionMemory,
} from '@/lib/medagent-api';
import { createConversationIdentity } from '@/lib/medical-conversation';

type Section = 'consultation' | 'history' | 'records' | 'profile' | 'settings';

const NAV_ITEMS = [
  { id: 'consultation', label: '医疗咨询', icon: StethoscopeIcon },
  { id: 'history', label: '历史会话', icon: Clock3Icon },
  { id: 'records', label: '病历 / 医学资料', icon: FileHeartIcon },
  { id: 'profile', label: '健康档案', icon: HeartPulseIcon },
  { id: 'settings', label: '设置', icon: SettingsIcon },
] satisfies Array<{ id: Section; label: string; icon: typeof StethoscopeIcon }>;

function LoginView({ onAuthenticated }: { onAuthenticated: (user: MedAgentUser) => void }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [pending, setPending] = useState(false);
  const [error, setError] = useState('');
  const [mode, setMode] = useState<'login' | 'register'>('login');

  async function submit(event: FormEvent) {
    event.preventDefault();
    setPending(true);
    setError('');
    try {
      await (mode === 'login' ? login : register)(username.trim(), password);
      onAuthenticated(await getCurrentUser());
    } catch (reason) {
      setError(
        reason instanceof Error ? reason.message : mode === 'login' ? '登录失败' : '注册失败'
      );
    } finally {
      setPending(false);
    }
  }

  return (
    <main className="medical-page-bg grid min-h-svh place-items-center px-4 py-8">
      <div className="grid w-full max-w-4xl overflow-hidden rounded-[28px] border border-white/60 bg-white shadow-[0_24px_80px_rgba(39,61,73,0.16)] md:grid-cols-[0.95fr_1.05fr]">
        <section className="medical-sidebar relative hidden overflow-hidden p-10 text-white md:flex md:flex-col">
          <div className="relative z-10 flex items-center gap-3">
            <div className="grid size-11 place-items-center rounded-2xl bg-white/12">
              <MedAgentLogo className="size-7 text-[#183c38]" />
            </div>
            <div>
              <p className="text-xl font-semibold tracking-tight">MedAgent</p>
              <p className="text-xs text-white/60">你的智能医疗助手</p>
            </div>
          </div>
          <div className="relative z-10 my-auto">
            <h1 className="max-w-sm text-3xl leading-tight font-semibold">
              让每一次健康咨询，更清楚、更安心
            </h1>
            <p className="mt-5 max-w-sm text-sm leading-7 text-white/68">
              结合你的健康档案与医学资料，支持文字和实时语音咨询，并为回答保留可查看的依据。
            </p>
          </div>
          <p className="relative z-10 flex items-center gap-2 text-xs text-white/55">
            <ShieldCheckIcon className="size-4" aria-hidden="true" />
            采用安全登录与隐私保护
          </p>
        </section>
        <form onSubmit={submit} className="p-7 sm:p-10 md:p-12">
          <div className="mb-8 flex items-center gap-3 md:hidden">
            <div className="grid size-10 place-items-center rounded-xl bg-[#e4edef] text-[#4f7184]">
              <MedAgentLogo className="size-6 text-[#183c38]" />
            </div>
            <div>
              <p className="font-semibold text-slate-800">MedAgent</p>
              <p className="text-xs text-slate-500">你的智能医疗助手</p>
            </div>
          </div>
          <p className="text-xs font-semibold tracking-[0.18em] text-[#6c8796]">WELCOME</p>
          <h2 className="mt-3 text-2xl font-semibold text-slate-800">
            {mode === 'login' ? '登录医疗助手' : '创建健康账户'}
          </h2>
          <p className="mt-2 text-sm leading-6 text-slate-500">
            {mode === 'login'
              ? '继续你的健康咨询与资料管理。'
              : '安全保存你的咨询、资料与健康信息。'}
          </p>
          <label className="mt-7 block text-sm font-medium text-slate-700">
            用户名
            <input
              required
              autoComplete="username"
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              className="mt-2 w-full rounded-xl border border-slate-200 bg-slate-50/70 px-4 py-3 transition outline-none focus:border-[#7594a5] focus:ring-4 focus:ring-[#7594a5]/12"
              placeholder="请输入用户名"
            />
          </label>
          <label className="mt-4 block text-sm font-medium text-slate-700">
            密码
            <input
              required
              type="password"
              autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              className="mt-2 w-full rounded-xl border border-slate-200 bg-slate-50/70 px-4 py-3 transition outline-none focus:border-[#7594a5] focus:ring-4 focus:ring-[#7594a5]/12"
              placeholder="请输入密码"
            />
          </label>
          {error && (
            <p role="alert" className="mt-4 text-sm text-rose-600">
              {error}
            </p>
          )}
          <button
            disabled={pending}
            className="mt-6 w-full rounded-xl bg-[#587a8e] px-4 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-[#496b7e] focus-visible:ring-2 focus-visible:ring-[#587a8e] focus-visible:ring-offset-2 focus-visible:outline-none disabled:opacity-50"
          >
            {pending ? '请稍候…' : mode === 'login' ? '安全登录' : '创建账户'}
          </button>
          <button
            type="button"
            className="mt-4 w-full rounded-lg py-2 text-center text-sm text-slate-500 underline-offset-4 hover:text-slate-700 hover:underline focus-visible:ring-2 focus-visible:ring-[#587a8e] focus-visible:outline-none"
            onClick={() => {
              setMode(mode === 'login' ? 'register' : 'login');
              setError('');
            }}
          >
            {mode === 'login' ? '没有账号？立即注册' : '已有账号？返回登录'}
          </button>
        </form>
      </div>
    </main>
  );
}

function HistoryView() {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [detail, setDetail] = useState<SessionDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const refresh = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      setSessions((await listSessions()).sessions);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '历史会话读取失败');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  async function openSession(sessionId: string) {
    setError('');
    try {
      setDetail(await getSession(sessionId));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '会话详情读取失败');
    }
  }

  return (
    <section className="h-full overflow-y-auto p-4 pb-24 md:p-8" aria-labelledby="history-title">
      <div className="mx-auto max-w-5xl">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 id="history-title" className="text-xl font-semibold text-slate-800">
              历史会话
            </h1>
            <p className="mt-1 text-sm text-slate-500">回看已保存的文字医疗咨询记录。</p>
          </div>
          <button
            type="button"
            onClick={() => void refresh()}
            aria-label="刷新历史会话"
            className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600 hover:bg-slate-50 focus-visible:ring-2 focus-visible:ring-[#587a8e] focus-visible:outline-none"
          >
            刷新
          </button>
        </div>
        <div className="mt-6 grid gap-4 lg:grid-cols-[320px_1fr]">
          <div className="grid content-start gap-2">
            {sessions.map((session) => (
              <button
                key={session.session_id}
                type="button"
                onClick={() => void openSession(session.session_id)}
                className={`rounded-2xl border p-4 text-left transition focus-visible:ring-2 focus-visible:ring-[#587a8e] focus-visible:outline-none ${
                  detail?.session_id === session.session_id
                    ? 'border-[#91a9b6] bg-[#eef4f6]'
                    : 'border-slate-200 bg-white hover:border-slate-300'
                }`}
              >
                <p className="truncate text-sm font-medium text-slate-700">
                  {session.session_id.startsWith('web_conv_') ? '医疗咨询' : '历史咨询'}
                </p>
                <p className="mt-1 text-xs text-slate-400">
                  {session.message_count} 条消息 ·{' '}
                  {new Date(session.updated_at).toLocaleString('zh-CN')}
                </p>
              </button>
            ))}
            {!loading && sessions.length === 0 && (
              <div className="rounded-2xl border border-dashed border-slate-300 p-8 text-center text-sm text-slate-400">
                暂无历史会话
              </div>
            )}
            {loading && (
              <p role="status" className="text-sm text-slate-400">
                正在读取记录…
              </p>
            )}
          </div>
          <div className="min-h-64 rounded-2xl border border-slate-200 bg-white p-4 md:p-6">
            {detail ? (
              <div className="grid gap-4">
                {detail.messages.map((message, index) => (
                  <div
                    key={`${message.type}-${index}`}
                    className={`max-w-[86%] rounded-2xl px-4 py-3 text-sm leading-6 ${
                      message.type === 'human'
                        ? 'ml-auto rounded-br-md bg-[#5c7f92] text-white'
                        : 'rounded-bl-md bg-slate-100 text-slate-700'
                    }`}
                  >
                    {message.content}
                  </div>
                ))}
              </div>
            ) : (
              <div className="grid min-h-56 place-items-center text-center text-sm text-slate-400">
                选择一条会话查看详情
              </div>
            )}
          </div>
        </div>
        {error && (
          <p role="alert" className="mt-4 text-sm text-rose-600">
            {error}
          </p>
        )}
      </div>
    </section>
  );
}

function RecordsView() {
  const [open, setOpen] = useState(false);
  return (
    <section className="h-full overflow-y-auto p-4 pb-24 md:p-8" aria-labelledby="records-title">
      <div className="mx-auto max-w-4xl">
        <h1 id="records-title" className="text-xl font-semibold text-slate-800">
          病历与医学资料
        </h1>
        <p className="mt-1 text-sm leading-6 text-slate-500">
          集中保存检查报告、病历摘要和可信医学资料。语音咨询会使用你当前选中的资料库。
        </p>
        <div className="mt-6 grid gap-4 sm:grid-cols-2">
          <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="grid size-10 place-items-center rounded-xl bg-[#eaf1f4] text-[#4f7184]">
              <FileHeartIcon className="size-5" aria-hidden="true" />
            </div>
            <h2 className="mt-4 font-semibold text-slate-700">我的病历资料</h2>
            <p className="mt-2 text-sm leading-6 text-slate-500">
              上传、查看和管理个人检查报告与就诊记录。
            </p>
            <button
              type="button"
              onClick={() => setOpen(true)}
              className="mt-5 rounded-xl bg-[#587a8e] px-4 py-2.5 text-sm font-medium text-white hover:bg-[#496b7e] focus-visible:ring-2 focus-visible:ring-[#587a8e] focus-visible:ring-offset-2 focus-visible:outline-none"
            >
              管理资料
            </button>
          </article>
          <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="grid size-10 place-items-center rounded-xl bg-emerald-50 text-emerald-700">
              <BookOpenTextIcon className="size-5" aria-hidden="true" />
            </div>
            <h2 className="mt-4 font-semibold text-slate-700">医学参考资料</h2>
            <p className="mt-2 text-sm leading-6 text-slate-500">
              整理咨询时需要参考的指南、说明和健康资料。
            </p>
            <button
              type="button"
              onClick={() => setOpen(true)}
              className="mt-5 rounded-xl border border-slate-200 px-4 py-2.5 text-sm font-medium text-slate-600 hover:bg-slate-50 focus-visible:ring-2 focus-visible:ring-[#587a8e] focus-visible:outline-none"
            >
              查看资料
            </button>
          </article>
        </div>
      </div>
      <LiveRagKnowledgePanel open={open} onOpenChange={setOpen} />
    </section>
  );
}

function HealthProfileView() {
  const [items, setItems] = useState<MedicalMemory[]>([]);
  const [error, setError] = useState('');
  const [pendingId, setPendingId] = useState('');

  const refresh = useCallback(async () => {
    try {
      setError('');
      setItems((await listMemories()).memories);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '健康档案读取失败');
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
        const content = window.prompt('请输入正确的健康信息', memory.content)?.trim();
        if (!content) return;
        await correctMemory(memory.memory_id, content);
      } else if (window.confirm('确认从健康档案中删除这条信息？')) {
        await deleteMemory(memory.memory_id);
      } else {
        return;
      }
      await refresh();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '健康档案操作失败');
    } finally {
      setPendingId('');
    }
  }

  const statusLabel: Record<MedicalMemory['status'], string> = {
    proposed: '待你确认',
    confirmed: '已确认',
    superseded: '已更新',
    rejected: '已忽略',
  };

  return (
    <section className="h-full overflow-y-auto p-4 pb-24 md:p-8" aria-labelledby="profile-title">
      <div className="mx-auto max-w-4xl">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h1 id="profile-title" className="text-xl font-semibold text-slate-800">
              我的健康档案
            </h1>
            <p className="mt-1 text-sm leading-6 text-slate-500">
              查看并管理助手记住的过敏史、既往情况和健康偏好；未经你确认的信息不会当作可靠事实使用。
            </p>
          </div>
          <button
            type="button"
            onClick={() => window.location.assign('/api/medagent/memories/export')}
            className="rounded-xl border border-slate-200 bg-white px-4 py-2 text-sm text-slate-600 hover:bg-slate-50 focus-visible:ring-2 focus-visible:ring-[#587a8e] focus-visible:outline-none"
          >
            导出健康信息
          </button>
        </div>
        <div className="mt-6 grid gap-3">
          {items.map((memory) => (
            <article
              key={memory.memory_id}
              className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm md:p-5"
            >
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="rounded-full bg-[#edf3f5] px-2.5 py-1 text-xs font-medium text-[#4f7184]">
                      {statusLabel[memory.status]}
                    </span>
                    <span className="text-xs text-slate-400">{memory.memory_type}</span>
                  </div>
                  <p className="mt-3 text-sm leading-6 text-slate-700">{memory.content}</p>
                </div>
                <div className="flex flex-wrap gap-2">
                  {memory.status === 'proposed' && (
                    <>
                      <button
                        disabled={pendingId === memory.memory_id}
                        onClick={() => void act(memory, 'confirm')}
                        className="rounded-lg bg-emerald-50 px-3 py-1.5 text-xs font-medium text-emerald-700 hover:bg-emerald-100 focus-visible:ring-2 focus-visible:ring-emerald-600 focus-visible:outline-none"
                        aria-label="确认这条健康信息"
                      >
                        确认
                      </button>
                      <button
                        disabled={pendingId === memory.memory_id}
                        onClick={() => void act(memory, 'reject')}
                        className="rounded-lg border border-slate-200 px-3 py-1.5 text-xs text-slate-600 hover:bg-slate-50 focus-visible:ring-2 focus-visible:ring-[#587a8e] focus-visible:outline-none"
                        aria-label="忽略这条健康信息"
                      >
                        忽略
                      </button>
                    </>
                  )}
                  <button
                    disabled={pendingId === memory.memory_id}
                    onClick={() => void act(memory, 'correct')}
                    className="rounded-lg border border-slate-200 px-3 py-1.5 text-xs text-slate-600 hover:bg-slate-50 focus-visible:ring-2 focus-visible:ring-[#587a8e] focus-visible:outline-none"
                    aria-label="纠正这条健康信息"
                  >
                    纠正
                  </button>
                  <button
                    disabled={pendingId === memory.memory_id}
                    onClick={() => void act(memory, 'delete')}
                    className="rounded-lg px-3 py-1.5 text-xs text-rose-600 hover:bg-rose-50 focus-visible:ring-2 focus-visible:ring-rose-600 focus-visible:outline-none"
                    aria-label="删除这条健康信息"
                  >
                    删除
                  </button>
                </div>
              </div>
            </article>
          ))}
          {items.length === 0 && !error && (
            <div className="rounded-2xl border border-dashed border-slate-300 p-10 text-center">
              <HeartPulseIcon className="mx-auto size-7 text-slate-300" aria-hidden="true" />
              <p className="mt-3 text-sm text-slate-400">暂无健康档案信息</p>
            </div>
          )}
          {error && (
            <p role="alert" className="text-sm text-rose-600">
              {error}
            </p>
          )}
        </div>
      </div>
    </section>
  );
}

function SettingsView() {
  const [open, setOpen] = useState(false);
  return (
    <section className="h-full overflow-y-auto p-4 pb-24 md:p-8" aria-labelledby="settings-title">
      <div className="mx-auto max-w-4xl">
        <h1 id="settings-title" className="text-xl font-semibold text-slate-800">
          设置
        </h1>
        <p className="mt-1 text-sm text-slate-500">管理咨询偏好与高级语音、资料检索配置。</p>
        <article className="mt-6 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-start gap-3">
            <div className="grid size-10 place-items-center rounded-xl bg-slate-100 text-slate-600">
              <SettingsIcon className="size-5" aria-hidden="true" />
            </div>
            <div className="flex-1">
              <h2 className="font-semibold text-slate-700">高级设置</h2>
              <p className="mt-1 text-sm leading-6 text-slate-500">
                调整语音识别、播报、资料检索和助手行为。通常无需修改。
              </p>
              <button
                type="button"
                onClick={() => setOpen(true)}
                className="mt-4 rounded-xl border border-slate-200 px-4 py-2 text-sm text-slate-600 hover:bg-slate-50 focus-visible:ring-2 focus-visible:ring-[#587a8e] focus-visible:outline-none"
              >
                打开高级设置
              </button>
            </div>
          </div>
        </article>
      </div>
      <LiveRagSettingsPanel open={open} onOpenChange={setOpen} />
    </section>
  );
}

export function UnifiedApp({ appConfig }: { appConfig: AppConfig }) {
  const [user, setUser] = useState<MedAgentUser | null>(null);
  const [checking, setChecking] = useState(true);
  const [section, setSection] = useState<Section>('consultation');
  const [mobileOpen, setMobileOpen] = useState(false);
  const [identity, setIdentity] = useState<ReturnType<typeof createConversationIdentity> | null>(
    null
  );

  useEffect(() => {
    setIdentity(createConversationIdentity(crypto.randomUUID()));
    getCurrentUser()
      .then(setUser)
      .catch(() => setUser(null))
      .finally(() => setChecking(false));
  }, []);

  const currentLabel = useMemo(
    () => NAV_ITEMS.find((item) => item.id === section)?.label ?? '医疗咨询',
    [section]
  );

  async function signOut() {
    await logout();
    setUser(null);
  }

  function navigate(next: Section) {
    setSection(next);
    setMobileOpen(false);
  }

  if (checking || !identity) {
    return (
      <main className="medical-page-bg grid min-h-svh place-items-center text-sm text-slate-500">
        <div role="status" className="flex items-center gap-3">
          <MedAgentLogo className="size-6 animate-pulse text-[#183c38]" />
          正在安全加载…
        </div>
      </main>
    );
  }
  if (!user) return <LoginView onAuthenticated={setUser} />;

  return (
    <div className="medical-page-bg flex h-svh overflow-hidden text-slate-700">
      <aside className="medical-sidebar hidden w-[252px] shrink-0 flex-col text-white md:flex">
        <div className="flex items-center gap-3 px-6 py-6">
          <div className="grid size-10 place-items-center rounded-2xl bg-white/12">
            <MedAgentLogo className="size-6 text-[#183c38]" />
          </div>
          <div>
            <p className="text-lg font-semibold tracking-tight">MedAgent</p>
            <p className="text-[11px] text-white/55">智能医疗助手</p>
          </div>
        </div>
        <nav className="mt-3 grid gap-1 px-3" aria-label="主要功能">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const selected = section === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => navigate(item.id)}
                aria-current={selected ? 'page' : undefined}
                className={`flex items-center gap-3 rounded-xl px-3 py-3 text-left text-sm transition focus-visible:ring-2 focus-visible:ring-white/70 focus-visible:outline-none ${
                  selected
                    ? 'bg-white text-[#324f60] shadow-sm'
                    : 'text-white/72 hover:bg-white/8 hover:text-white'
                }`}
              >
                <Icon className="size-[18px]" aria-hidden="true" />
                {item.label}
              </button>
            );
          })}
        </nav>
        <div className="mt-auto px-4 pb-5">
          <div className="rounded-2xl border border-white/10 bg-white/6 p-3">
            <div className="flex items-center gap-2.5">
              <div className="grid size-8 place-items-center rounded-full bg-white/12">
                <UserRoundIcon className="size-4" aria-hidden="true" />
              </div>
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium">{user.username}</p>
                <p className="text-[10px] text-white/45">健康账户</p>
              </div>
              <button
                type="button"
                onClick={() => void signOut()}
                aria-label="退出登录"
                className="rounded-lg p-2 text-white/55 hover:bg-white/10 hover:text-white focus-visible:ring-2 focus-visible:ring-white/70 focus-visible:outline-none"
              >
                <LogOutIcon className="size-4" aria-hidden="true" />
              </button>
            </div>
          </div>
          <p className="mt-4 flex items-start gap-2 px-1 text-[10px] leading-4 text-white/42">
            <ShieldCheckIcon className="mt-0.5 size-3 shrink-0" aria-hidden="true" />
            医疗建议仅供参考，紧急情况请及时就医
          </p>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-14 shrink-0 items-center border-b border-slate-200/80 bg-white px-4 md:hidden">
          <button
            type="button"
            onClick={() => setMobileOpen(true)}
            aria-label="打开功能菜单"
            className="grid size-9 place-items-center rounded-lg text-slate-600 hover:bg-slate-100 focus-visible:ring-2 focus-visible:ring-[#587a8e] focus-visible:outline-none"
          >
            <MenuIcon className="size-5" aria-hidden="true" />
          </button>
          <MedAgentLogo className="ml-3 size-5 text-[#183c38]" />
          <span className="ml-2 font-semibold text-slate-800">MedAgent</span>
          <span className="ml-auto text-xs text-slate-400">{currentLabel}</span>
        </header>

        <main className="relative flex min-h-0 flex-1 flex-col">
          <MedicalConsultation
            appConfig={appConfig}
            identity={identity}
            active={section === 'consultation'}
          />
          {section === 'history' && <HistoryView />}
          {section === 'records' && <RecordsView />}
          {section === 'profile' && <HealthProfileView />}
          {section === 'settings' && <SettingsView />}
        </main>

        <nav
          className="grid h-[66px] shrink-0 grid-cols-4 border-t border-slate-200 bg-white pb-[env(safe-area-inset-bottom)] md:hidden"
          aria-label="移动端主要功能"
        >
          {NAV_ITEMS.slice(0, 4).map((item) => {
            const Icon = item.icon;
            const selected = section === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => navigate(item.id)}
                aria-current={selected ? 'page' : undefined}
                className={`flex flex-col items-center justify-center gap-1 text-[10px] focus-visible:ring-2 focus-visible:ring-[#587a8e] focus-visible:outline-none focus-visible:ring-inset ${
                  selected ? 'text-[#496b7e]' : 'text-slate-400'
                }`}
              >
                <Icon className="size-5" aria-hidden="true" />
                {item.id === 'records' ? '医学资料' : item.label}
              </button>
            );
          })}
        </nav>
      </div>

      {mobileOpen && (
        <div className="fixed inset-0 z-[200] bg-slate-950/30 backdrop-blur-sm md:hidden">
          <button
            type="button"
            aria-label="关闭功能菜单"
            className="absolute inset-0 size-full cursor-default"
            onClick={() => setMobileOpen(false)}
          />
          <aside className="medical-sidebar relative flex h-full w-[82%] max-w-xs flex-col p-4 text-white shadow-2xl">
            <div className="flex items-center gap-3 px-2 py-3">
              <div className="grid size-10 place-items-center rounded-xl bg-white/12">
                <MedAgentLogo className="size-6 text-[#183c38]" />
              </div>
              <div>
                <p className="font-semibold">MedAgent</p>
                <p className="text-xs text-white/55">{user.username}</p>
              </div>
              <button
                type="button"
                onClick={() => setMobileOpen(false)}
                aria-label="关闭菜单"
                className="ml-auto grid size-9 place-items-center rounded-lg hover:bg-white/10 focus-visible:ring-2 focus-visible:ring-white focus-visible:outline-none"
              >
                <XIcon className="size-5" aria-hidden="true" />
              </button>
            </div>
            <nav className="mt-5 grid gap-1" aria-label="全部功能">
              {NAV_ITEMS.map((item) => {
                const Icon = item.icon;
                return (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => navigate(item.id)}
                    className={`flex items-center gap-3 rounded-xl px-3 py-3 text-sm focus-visible:ring-2 focus-visible:ring-white focus-visible:outline-none ${
                      section === item.id
                        ? 'bg-white text-[#324f60]'
                        : 'text-white/75 hover:bg-white/10'
                    }`}
                  >
                    <Icon className="size-5" aria-hidden="true" />
                    {item.label}
                  </button>
                );
              })}
            </nav>
            <button
              type="button"
              onClick={() => void signOut()}
              className="mt-auto flex items-center gap-3 rounded-xl px-3 py-3 text-sm text-white/70 hover:bg-white/10 focus-visible:ring-2 focus-visible:ring-white focus-visible:outline-none"
            >
              <LogOutIcon className="size-5" aria-hidden="true" />
              退出登录
            </button>
          </aside>
        </div>
      )}
    </div>
  );
}
