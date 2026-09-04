'use client';

import { useEffect, useMemo, useState } from 'react';
import { createPortal } from 'react-dom';
import { DatabaseIcon, FileTextIcon, XIcon } from 'lucide-react';
import {
  type LiveRagEvidenceChunk,
  type LiveRagEvidenceDocument,
  type LiveRagRuntimeState,
  type LiveRagTurn,
} from '@/lib/liverag-api';
import { decodeLiveRagDisplayText, getLiveRagDisplayName } from '@/lib/liverag-display';
import { cn } from '@/lib/shadcn/utils';

interface LiveRagEvidencePanelProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  turns: LiveRagTurn[];
  runtime: LiveRagRuntimeState | null;
}

type EvidenceSource =
  | { type: 'chunk'; item: LiveRagEvidenceChunk }
  | { type: 'document'; item: LiveRagEvidenceDocument };

function getLatestRagTurn(turns: LiveRagTurn[]) {
  return [...turns]
    .reverse()
    .find((turn) => turn.rag && turn.rag.status && turn.rag.status !== 'not_queried');
}

function sourceTitle(source: EvidenceSource) {
  const item = source.item;
  if (item.file_path) return getLiveRagDisplayName(item.file_path);
  if ('title' in item && item.title) return decodeLiveRagDisplayText(item.title);
  return decodeLiveRagDisplayText(item.document_id) || '知识库片段';
}

function sourceKey(source: EvidenceSource, index: number) {
  const item = source.item;
  const identity =
    [
      source.type === 'chunk' ? source.item.chunk_id : undefined,
      item.document_id,
      item.file_path,
      source.type === 'document' ? source.item.title : undefined,
    ].find((value) => value && value.trim()) ?? 'unknown';

  return `${source.type}-${index}-${identity}`;
}

function sourceKnowledgeName(source: EvidenceSource) {
  return source.item.kb_name ?? '当前知识库';
}

function scoreLabel(score?: number) {
  return typeof score === 'number' ? `score ${score.toFixed(2)}` : null;
}

function getTurnSources(turn?: LiveRagTurn): EvidenceSource[] {
  const rag = turn?.rag;
  if (!rag) return [];

  const chunks = rag.evidence_chunks ?? [];
  if (chunks.length > 0) {
    return chunks.map((item) => ({ type: 'chunk', item }));
  }

  return (rag.evidence_documents ?? []).map((item) => ({ type: 'document', item }));
}

function getRecentDocuments(turns: LiveRagTurn[]) {
  const seen = new Set<string>();
  const documents: EvidenceSource[] = [];

  for (const turn of [...turns].reverse()) {
    for (const source of getTurnSources(turn)) {
      const item = source.item;
      const key = item.document_id ?? item.file_path;
      if (!key || seen.has(key)) continue;
      seen.add(key);
      documents.push(source);
      if (documents.length >= 6) return documents;
    }
  }

  return documents;
}

function scopeLabel(runtime: LiveRagRuntimeState | null) {
  const knowledgeBase = runtime?.knowledge_base;
  const active = knowledgeBase?.active_session;
  const configured = knowledgeBase?.configured;

  if (active?.name || active?.kb_id) {
    return `${active.name ?? active.kb_id}${knowledgeBase?.locked ? ' · 已锁定' : ''}`;
  }

  if (configured?.name || configured?.kb_id) {
    return `${configured.name ?? configured.kb_id} · 下次通话`;
  }

  return '默认知识库';
}

function EvidenceCard({ source }: { source: EvidenceSource }) {
  const preview =
    source.type === 'chunk'
      ? (source.item.content_preview ?? '暂无命中文本摘要')
      : `${source.item.chunk_count ?? 0} 个相关片段`;
  const score = source.type === 'chunk' ? scoreLabel(source.item.score) : null;

  return (
    <article className="bg-muted/35 grid gap-2 rounded-xl p-3">
      <div className="flex min-w-0 items-center gap-2">
        <span className="bg-background grid size-8 shrink-0 place-items-center rounded-lg">
          <FileTextIcon className="size-4" />
        </span>
        <div className="min-w-0">
          <h3 className="truncate text-sm font-semibold">{sourceTitle(source)}</h3>
          <p className="text-muted-foreground truncate text-[11px]">
            {sourceKnowledgeName(source)}
            {score ? ` · ${score}` : ''}
          </p>
        </div>
      </div>
      <p className="text-muted-foreground line-clamp-2 text-xs leading-relaxed">{preview}</p>
    </article>
  );
}

export function LiveRagEvidencePanel({
  open,
  onOpenChange,
  turns,
  runtime,
}: LiveRagEvidencePanelProps) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const latestTurn = useMemo(() => getLatestRagTurn(turns), [turns]);
  const latestRag = latestTurn?.rag;
  const latestSources = useMemo(() => getTurnSources(latestTurn), [latestTurn]);
  const recentDocuments = useMemo(() => getRecentDocuments(turns), [turns]);
  const hasLatestEvidence = latestSources.length > 0;

  if (!mounted) return null;

  return createPortal(
    <div
      className={cn(
        'fixed inset-0 z-[999] flex items-end justify-center p-3 transition-[visibility] sm:justify-end sm:p-6',
        open ? 'visible' : 'invisible'
      )}
      aria-hidden={!open}
    >
      <button
        type="button"
        aria-label="关闭回答依据面板"
        className={cn(
          'absolute inset-0 cursor-default bg-transparent transition-opacity',
          open ? 'opacity-100' : 'pointer-events-none opacity-0'
        )}
        onClick={() => onOpenChange(false)}
      />
      <aside
        role="dialog"
        aria-modal="true"
        aria-label="回答依据"
        className={cn(
          'bg-background/95 relative grid h-[min(560px,58vh)] w-full max-w-[480px] grid-rows-[auto_minmax(0,1fr)] overflow-hidden rounded-2xl border shadow-[0_24px_80px_rgba(0,0,0,0.18)] backdrop-blur-xl transition duration-200 ease-out',
          open ? 'translate-y-0 scale-100 opacity-100' : 'translate-y-4 scale-[0.98] opacity-0'
        )}
      >
        <header className="flex items-start justify-between gap-3 border-b px-4 py-3.5">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <DatabaseIcon className="size-4" />
              <h2 className="truncate text-base font-semibold">回答依据</h2>
            </div>
            <p className="text-muted-foreground mt-1 text-xs">这里展示当前回答实际引用的资料。</p>
          </div>
          <button
            type="button"
            onClick={() => onOpenChange(false)}
            className="hover:bg-muted grid size-8 shrink-0 place-items-center rounded-full border"
            aria-label="关闭"
          >
            <XIcon className="size-4" />
          </button>
        </header>

        <div className="min-h-0 overflow-auto p-4">
          <section className="bg-muted/30 mb-4 rounded-2xl p-3">
            <div className="text-muted-foreground text-[11px] font-semibold tracking-wider uppercase">
              当前资料库
            </div>
            <div className="mt-1 truncate text-sm font-semibold">{scopeLabel(runtime)}</div>
          </section>

          <section className="grid gap-3">
            <div className="flex items-center justify-between gap-3">
              <h3 className="text-sm font-semibold">本次回答依据</h3>
              {latestRag?.status && latestRag.status !== 'hit' && (
                <span className="text-muted-foreground text-xs">
                  {latestRag.status === 'miss' ? '未命中' : '查询失败'}
                </span>
              )}
            </div>

            {hasLatestEvidence ? (
              <div className="grid gap-2">
                {latestSources.slice(0, 5).map((source, index) => (
                  <EvidenceCard key={sourceKey(source, index)} source={source} />
                ))}
              </div>
            ) : (
              <div className="border-border/60 grid min-h-28 place-items-center rounded-2xl border border-dashed p-4 text-center">
                <div>
                  <div className="text-sm font-semibold">
                    {latestRag?.status === 'miss' ? '本轮未找到相关资料' : '当前还没有引用医学资料'}
                  </div>
                  <p className="text-muted-foreground mt-1 text-xs">
                    回答引用资料后，这里会显示对应文档和片段。
                  </p>
                </div>
              </div>
            )}
          </section>

          {recentDocuments.length > 0 && (
            <section className="mt-5 grid gap-3">
              <h3 className="text-sm font-semibold">最近引用过的文档</h3>
              <div className="grid gap-1">
                {recentDocuments.map((source, index) => (
                  <div
                    key={sourceKey(source, index)}
                    className="text-muted-foreground flex min-w-0 items-center justify-between gap-3 rounded-lg px-2 py-1.5 text-xs"
                  >
                    <span className="truncate">{sourceTitle(source)}</span>
                    <span className="shrink-0">{sourceKnowledgeName(source)}</span>
                  </div>
                ))}
              </div>
            </section>
          )}
        </div>
      </aside>
    </div>,
    document.body
  );
}
