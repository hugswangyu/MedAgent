'use client';

import { type ChangeEvent, useEffect, useMemo, useState } from 'react';
import { createPortal } from 'react-dom';
import {
  CheckIcon,
  DatabaseIcon,
  DownloadIcon,
  ExternalLinkIcon,
  FilePlus2Icon,
  FileTextIcon,
  FolderIcon,
  LayoutGridIcon,
  ListIcon,
  Loader2Icon,
  PencilIcon,
  PlusIcon,
  RefreshCwIcon,
  SearchIcon,
  Trash2Icon,
  UploadIcon,
  XIcon,
} from 'lucide-react';
import { MessageResponse } from '@/components/ai-elements/message';
import {
  LiveRagApiError,
  type LiveRagDocument,
  type LiveRagDocumentsResponse,
  type LiveRagKnowledgeBase,
  type LiveRagReady,
  type LiveRagSessionKnowledgeBase,
  clearKnowledgeBaseDocuments,
  createKnowledgeBase,
  deleteKnowledgeBase,
  deleteKnowledgeBaseDocument,
  getKnowledgeBaseDocumentSourceUrl,
  getKnowledgeBaseDocuments,
  getKnowledgeBaseJob,
  getKnowledgeBaseReady,
  getKnowledgeBases,
  getRagReady,
  getSessionKnowledgeBase,
  setSessionKnowledgeBase,
  updateKnowledgeBase,
  uploadKnowledgeBaseFiles,
  uploadKnowledgeBaseText,
} from '@/lib/liverag-api';
import { getLiveRagDisplayName } from '@/lib/liverag-display';
import { cn } from '@/lib/shadcn/utils';

interface LiveRagKnowledgePanelProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

function formatDate(value?: string) {
  if (!value) return '未记录';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleDateString(undefined, { month: '2-digit', day: '2-digit' });
}

function statusLabel(status?: string) {
  if (!status) return '未知';
  if (status === 'processed') return '已索引';
  if (status === 'failed') return '失败';
  if (status === 'parse_failed') return '解析失败';
  if (status === 'pending') return '等待中';
  if (status === 'processing') return '索引中';
  return status;
}

function getDocumentKey(document: LiveRagDocument) {
  return document.document_id ?? document.file_path ?? document.track_id ?? 'unknown-document';
}

function getDocumentTitle(document: LiveRagDocument) {
  const raw =
    document.original_filename ?? document.file_path ?? document.document_id ?? '未命名文档';
  return getLiveRagDisplayName(raw) || raw;
}

function getDocumentExtension(document: LiveRagDocument) {
  if (document.extension) return document.extension.replace(/^\./, '').slice(0, 5).toUpperCase();
  const title = getDocumentTitle(document);
  const extension = title.includes('.') ? title.split('.').pop() : undefined;
  return extension?.slice(0, 5).toUpperCase() ?? 'DOC';
}

function getDocumentType(document: LiveRagDocument) {
  const extension = getDocumentExtension(document).toLowerCase();
  if (extension === 'doc' || extension === 'docx') return 'Microsoft Word';
  if (extension === 'pdf') return 'PDF 文档';
  if (extension === 'md') return 'Markdown';
  if (extension === 'txt') return '文本';
  return `${getDocumentExtension(document)} 文件`;
}

function canPreviewDocument(document: LiveRagDocument) {
  const contentType = document.content_type?.toLowerCase() ?? '';
  const extension = getDocumentExtension(document).toLowerCase();
  return (
    contentType.startsWith('text/') ||
    contentType.startsWith('image/') ||
    contentType === 'application/pdf' ||
    [
      'pdf',
      'txt',
      'md',
      'json',
      'csv',
      'html',
      'htm',
      'png',
      'jpg',
      'jpeg',
      'gif',
      'webp',
    ].includes(extension)
  );
}

function canOpenDocumentSource(document: LiveRagDocument) {
  return Boolean(document.document_id && document.source_file_exists !== false);
}

function isMarkdownDocument(document: LiveRagDocument) {
  const contentType = document.content_type?.toLowerCase() ?? '';
  const extension = getDocumentExtension(document).toLowerCase();
  return contentType === 'text/markdown' || extension === 'md' || extension === 'markdown';
}

function getDocumentMeta(document: LiveRagDocument) {
  const parts = [];
  if (typeof document.chunks_count === 'number') parts.push(`${document.chunks_count} 个片段`);
  if (typeof document.content_length === 'number') parts.push(`${document.content_length} 字符`);
  if (document.parse_status === 'failed') parts.push('解析失败');
  else if (document.index_status === 'processing') parts.push('索引中');
  parts.push(formatDate(document.updated_at ?? document.created_at));
  return parts.join(' · ');
}

function getDocumentSize(document: LiveRagDocument) {
  const length = document.source_file_size ?? document.content_length;
  if (typeof length !== 'number' || length <= 0) return '--';
  if (length >= 1024 * 1024) return `${(length / 1024 / 1024).toFixed(1)} MB`;
  if (length >= 1024) return `${Math.round(length / 1024)} KB`;
  return `${length} B`;
}

function getDocumentPreview(document: LiveRagDocument) {
  return (
    document.content_summary ??
    document.content ??
    `${document.chunks_count ?? 0} 个索引片段，可在语音对话中作为回答依据。`
  )
    .replace(/\s+/g, ' ')
    .trim();
}

function getDocumentPreviewLines(document: LiveRagDocument) {
  const preview = getDocumentPreview(document);
  return preview ? [preview.slice(0, 28), preview.slice(28, 56), preview.slice(56, 84)] : [];
}

function errorMessage(err: unknown, fallback: string) {
  if (err instanceof LiveRagApiError) {
    if (err.status === 409 || err.type === 'KnowledgeBaseLocked') {
      return '当前通话正在使用该知识库，请挂断后再操作。';
    }
    if (err.status === 404) return '知识库或文档不存在，请刷新后重试。';
    return err.message || fallback;
  }
  return err instanceof Error ? err.message : fallback;
}

function delay(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function StatusPill({ status }: { status?: string }) {
  const isReady = status === 'processed';
  const isFailed = status === 'failed' || status === 'parse_failed';

  return (
    <span
      className={cn(
        'inline-flex shrink-0 rounded-full px-2.5 py-1 text-[11px] whitespace-nowrap',
        isReady && 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-300',
        isFailed && 'bg-destructive/10 text-destructive',
        !isReady && !isFailed && 'bg-muted text-muted-foreground'
      )}
    >
      {statusLabel(status)}
    </span>
  );
}

function KnowledgeBaseButton({
  knowledgeBase,
  active,
  configured,
  locked,
  onClick,
  onDelete,
  pendingDelete,
}: {
  knowledgeBase: LiveRagKnowledgeBase;
  active: boolean;
  configured: boolean;
  locked: boolean;
  onClick: () => void;
  onDelete: () => void;
  pendingDelete: boolean;
}) {
  const isDefault = knowledgeBase.kb_id === 'default';

  return (
    <div
      className={cn(
        'group flex items-center gap-2 rounded-xl px-2.5 py-2 text-sm transition',
        active
          ? 'bg-foreground text-background shadow-sm'
          : 'text-foreground hover:bg-background/70'
      )}
    >
      <button type="button" onClick={onClick} className="flex min-w-0 flex-1 items-center gap-3">
        <span
          className={cn(
            'grid size-8 shrink-0 place-items-center rounded-lg',
            active ? 'bg-background/15' : 'bg-background/75'
          )}
        >
          <FolderIcon className="size-4" />
        </span>
        <span className="min-w-0 text-left">
          <span className="flex min-w-0 items-center gap-1.5">
            <span className="truncate font-medium">{knowledgeBase.name}</span>
            {configured && (
              <span className="shrink-0 rounded-full bg-emerald-500/15 px-1.5 py-0.5 text-[10px] text-emerald-700 dark:text-emerald-300">
                当前
              </span>
            )}
          </span>
          <span
            className={cn(
              'block truncate text-xs',
              active ? 'text-background/65' : 'text-muted-foreground'
            )}
          >
            {knowledgeBase.document_count ?? 0} 个文件 · {knowledgeBase.chunk_count ?? 0} 个片段
          </span>
        </span>
      </button>
      {!isDefault && (
        <button
          type="button"
          aria-label={`删除知识库 ${knowledgeBase.name}`}
          onClick={onDelete}
          disabled={locked}
          className={cn(
            'grid size-7 shrink-0 place-items-center rounded-full opacity-0 transition group-hover:opacity-100 disabled:cursor-not-allowed disabled:opacity-30',
            pendingDelete
              ? 'bg-destructive text-destructive-foreground opacity-100'
              : active
                ? 'hover:bg-background/15'
                : 'hover:bg-destructive/10 hover:text-destructive'
          )}
          title={pendingDelete ? '再次点击确认删除' : '删除知识库'}
        >
          <Trash2Icon className="size-3.5" />
        </button>
      )}
    </div>
  );
}

function KnowledgeBaseChip({
  knowledgeBase,
  active,
  configured,
  onClick,
}: {
  knowledgeBase: LiveRagKnowledgeBase;
  active: boolean;
  configured: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'inline-flex h-9 max-w-56 shrink-0 items-center gap-2 overflow-hidden rounded-full border px-3 text-xs font-semibold transition',
        active
          ? 'bg-foreground text-background border-foreground'
          : 'bg-background/75 hover:bg-background text-foreground'
      )}
    >
      <FolderIcon className="size-3.5 shrink-0" />
      <span className="truncate">{knowledgeBase.name}</span>
      {configured && (
        <span className={cn('shrink-0', active ? 'text-background/75' : 'text-emerald-600')}>
          当前
        </span>
      )}
    </button>
  );
}

function DocumentRow({
  document,
  canOpenSource,
  onPreview,
  onDelete,
}: {
  document: LiveRagDocument;
  canOpenSource: boolean;
  onPreview: () => void;
  onDelete: () => void;
}) {
  const title = getDocumentTitle(document);

  return (
    <article
      role={canOpenSource ? 'button' : undefined}
      tabIndex={canOpenSource ? 0 : -1}
      onClick={canOpenSource ? onPreview : undefined}
      onKeyDown={(event) => {
        if (!canOpenSource) return;
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          onPreview();
        }
      }}
      className={cn(
        'hover:bg-muted/55 odd:bg-muted/25 grid min-h-9 grid-cols-[minmax(0,1fr)_36px] items-center gap-3 rounded-lg px-2.5 py-1 text-sm transition sm:grid-cols-[minmax(0,1fr)_132px_72px_104px_36px]',
        canOpenSource ? 'cursor-pointer' : 'cursor-default'
      )}
    >
      <div className="flex min-w-0 items-center gap-2">
        <span className="grid size-6 shrink-0 place-items-center rounded-md bg-blue-500/10 text-blue-600 dark:text-blue-300">
          <FileTextIcon className="size-3.5" />
        </span>
        <span className="truncate font-medium">{title}</span>
      </div>
      <span className="text-muted-foreground hidden truncate text-right text-xs sm:block">
        {formatDate(document.updated_at ?? document.created_at)}
      </span>
      <span className="text-muted-foreground hidden truncate text-right text-xs sm:block">
        {getDocumentSize(document)}
      </span>
      <span className="text-muted-foreground hidden truncate text-xs sm:block">
        {getDocumentType(document)}
      </span>
      <div className="flex items-center justify-end gap-1">
        <button
          type="button"
          aria-label="删除文档"
          className="hover:bg-destructive/10 hover:text-destructive grid size-7 place-items-center rounded-full transition disabled:opacity-40"
          disabled={!document.document_id}
          onClick={(event) => {
            event.stopPropagation();
            onDelete();
          }}
        >
          <Trash2Icon className="size-3.5" />
        </button>
      </div>
    </article>
  );
}

function DocumentCard({
  document,
  sourceUrl,
  canOpenSource,
  onPreview,
  onDelete,
}: {
  document: LiveRagDocument;
  sourceUrl?: string;
  canOpenSource: boolean;
  onPreview: () => void;
  onDelete: () => void;
}) {
  const title = getDocumentTitle(document);
  const extension = getDocumentExtension(document);
  const preview = getDocumentPreview(document);
  const previewLines = getDocumentPreviewLines(document);
  const contentType = document.content_type?.toLowerCase() ?? '';
  const isImagePreview = Boolean(sourceUrl && contentType.startsWith('image/'));
  const hasInlinePreview = Boolean(sourceUrl && canPreviewDocument(document));

  return (
    <article
      role={canOpenSource ? 'button' : undefined}
      tabIndex={canOpenSource ? 0 : -1}
      onClick={canOpenSource ? onPreview : undefined}
      onKeyDown={(event) => {
        if (!canOpenSource) return;
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          onPreview();
        }
      }}
      className={cn(
        'group bg-background/85 ring-border/40 grid min-h-[232px] grid-rows-[96px_minmax(0,1fr)] overflow-hidden rounded-xl shadow-sm ring-1 transition hover:shadow-[0_14px_34px_rgba(0,0,0,0.08)]',
        canOpenSource ? 'cursor-pointer hover:-translate-y-0.5' : 'cursor-default'
      )}
    >
      <div className="bg-muted/55 relative m-2 mb-0 overflow-hidden rounded-lg">
        {hasInlinePreview ? (
          isImagePreview ? (
            <img src={sourceUrl} alt="" className="h-full w-full object-cover" loading="lazy" />
          ) : (
            <iframe
              src={sourceUrl}
              title={`${title} 预览`}
              className="bg-background pointer-events-none absolute top-0 left-0 h-[220%] w-[220%] origin-top-left scale-[0.46] border-0"
              tabIndex={-1}
            />
          )
        ) : (
          <>
            <div className="absolute inset-0 bg-[linear-gradient(135deg,transparent,rgba(0,0,0,0.04))]" />
            <div className="bg-background/95 ring-border/40 absolute top-1/2 left-1/2 w-[76%] -translate-x-1/2 -translate-y-1/2 rounded-lg p-2 shadow-sm ring-1">
              <div className="mb-1.5 flex items-center justify-between gap-2">
                <div className="bg-muted/60 grid size-6 place-items-center rounded-lg">
                  <FileTextIcon className="size-3.5" />
                </div>
                <span className="text-muted-foreground font-mono text-[10px] font-bold tracking-wider">
                  {extension}
                </span>
              </div>
              <div className="grid gap-1">
                {(previewLines.length ? previewLines : ['知识库文档预览', '等待索引内容摘要']).map(
                  (line, index) => (
                    <div
                      key={`${line}-${index}`}
                      className={cn(
                        'bg-foreground/10 h-1.5 rounded-full',
                        index === 0 && 'w-full',
                        index === 1 && 'w-4/5',
                        index === 2 && 'w-3/5'
                      )}
                    />
                  )
                )}
              </div>
            </div>
          </>
        )}
        {hasInlinePreview && (
          <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(180deg,transparent_55%,rgba(0,0,0,0.06))]" />
        )}
        <div className="bg-background/80 text-muted-foreground absolute top-2 right-2 rounded-md px-2 py-1 font-mono text-[10px] font-bold tracking-wider backdrop-blur">
          {extension}
        </div>
        <div className="absolute right-2 bottom-2">
          <StatusPill status={document.status} />
        </div>
      </div>

      <div className="grid grid-rows-[auto_auto_auto] gap-1.5 p-2.5">
        <div className="min-w-0">
          <h3 className="line-clamp-2 min-h-8 text-[13px] leading-snug font-semibold">{title}</h3>
          <p className="text-muted-foreground mt-1 line-clamp-2 min-h-8 text-[11px] leading-relaxed">
            {preview}
          </p>
        </div>
        <div className="text-muted-foreground min-w-0 truncate text-[10px]">
          {getDocumentMeta(document)}
        </div>
        <div className="flex items-center justify-between gap-2">
          <span className="text-muted-foreground truncate text-[10px]">
            {document.kb_name ?? '当前知识库'}
          </span>
          <div className="flex items-center gap-1">
            <button
              type="button"
              aria-label="删除文档"
              className="hover:bg-destructive/10 hover:text-destructive grid size-7 shrink-0 place-items-center rounded-full transition disabled:opacity-40"
              disabled={!document.document_id}
              onClick={(event) => {
                event.stopPropagation();
                onDelete();
              }}
            >
              <Trash2Icon className="size-3.5" />
            </button>
          </div>
        </div>
      </div>
    </article>
  );
}

function DocumentPreviewPanel({
  kbId,
  document,
  onClose,
}: {
  kbId: string | null;
  document: LiveRagDocument | null;
  onClose: () => void;
}) {
  const [markdownContent, setMarkdownContent] = useState('');
  const [markdownLoading, setMarkdownLoading] = useState(false);
  const [markdownError, setMarkdownError] = useState<string | null>(null);

  const title = document ? getDocumentTitle(document) : '';
  const inlineUrl =
    kbId && document?.document_id
      ? getKnowledgeBaseDocumentSourceUrl(kbId, document.document_id, 'inline')
      : '';
  const downloadUrl =
    kbId && document?.document_id
      ? getKnowledgeBaseDocumentSourceUrl(kbId, document.document_id, 'attachment')
      : '';
  const previewable = document ? canPreviewDocument(document) : false;
  const markdown = document ? isMarkdownDocument(document) : false;

  useEffect(() => {
    if (!inlineUrl || !markdown) {
      setMarkdownContent('');
      setMarkdownError(null);
      setMarkdownLoading(false);
      return;
    }

    let cancelled = false;
    setMarkdownContent('');
    setMarkdownError(null);
    setMarkdownLoading(true);

    fetch(inlineUrl, { cache: 'no-store' })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`文件预览请求失败：${response.status}`);
        }
        return response.text();
      })
      .then((text) => {
        if (!cancelled) setMarkdownContent(text);
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setMarkdownError(err instanceof Error ? err.message : 'Markdown 文件读取失败');
        }
      })
      .finally(() => {
        if (!cancelled) setMarkdownLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [inlineUrl, markdown]);

  if (!kbId || !document?.document_id) return null;

  return (
    <div className="fixed inset-0 z-[1001] flex items-center justify-center bg-black/35 p-0 backdrop-blur-sm sm:p-6">
      <button
        type="button"
        aria-label="关闭文件预览"
        className="absolute inset-0 cursor-default"
        onClick={onClose}
      />
      <aside
        role="dialog"
        aria-modal="true"
        aria-label="文件预览"
        className="bg-background relative grid h-svh w-full max-w-none grid-rows-[auto_minmax(0,1fr)] overflow-hidden border-0 shadow-[0_28px_90px_rgba(0,0,0,0.2)] sm:h-[min(760px,calc(100vh-48px))] sm:max-w-[980px] sm:rounded-2xl sm:border"
      >
        <header className="flex items-center justify-between gap-3 border-b px-4 py-3">
          <div className="min-w-0">
            <div className="flex min-w-0 items-center gap-2">
              <FileTextIcon className="size-4 shrink-0" />
              <h2 className="truncate text-sm font-semibold">{title}</h2>
            </div>
            <p className="text-muted-foreground mt-0.5 truncate text-[11px]">
              {getDocumentType(document)} · {getDocumentSize(document)}
            </p>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <a
              href={inlineUrl}
              target="_blank"
              rel="noreferrer"
              className="hover:bg-muted grid size-8 place-items-center rounded-full border"
              aria-label="新标签页打开"
              title="新标签页打开"
            >
              <ExternalLinkIcon className="size-4" />
            </a>
            <a
              href={downloadUrl}
              className="hover:bg-muted grid size-8 place-items-center rounded-full border"
              aria-label="下载原文件"
              title="下载原文件"
            >
              <DownloadIcon className="size-4" />
            </a>
            <button
              type="button"
              onClick={onClose}
              className="hover:bg-muted grid size-8 place-items-center rounded-full border"
              aria-label="关闭"
            >
              <XIcon className="size-4" />
            </button>
          </div>
        </header>

        {markdown ? (
          <div className="bg-muted/25 min-h-0 overflow-auto px-4 py-5 sm:px-8">
            <div className="bg-background/85 ring-border/40 mx-auto min-h-full max-w-3xl rounded-2xl p-5 shadow-sm ring-1 sm:p-7">
              {markdownLoading ? (
                <div className="grid min-h-72 place-items-center">
                  <Loader2Icon className="text-muted-foreground size-7 animate-spin" />
                </div>
              ) : markdownError ? (
                <div className="grid min-h-72 place-items-center text-center">
                  <div className="grid max-w-sm justify-items-center gap-3">
                    <FileTextIcon className="text-muted-foreground size-10" />
                    <div>
                      <div className="text-sm font-semibold">Markdown 文件读取失败</div>
                      <p className="text-muted-foreground mt-1 text-xs leading-relaxed">
                        {markdownError}
                      </p>
                    </div>
                  </div>
                </div>
              ) : (
                <MessageResponse className="[&_code]:bg-muted [&_pre]:bg-muted text-sm leading-7 [&_a]:text-blue-600 [&_a]:underline dark:[&_a]:text-blue-300 [&_blockquote]:border-l-2 [&_blockquote]:pl-3 [&_code]:rounded [&_code]:px-1 [&_h1]:text-xl [&_h1]:font-semibold [&_h2]:text-lg [&_h2]:font-semibold [&_h3]:text-base [&_h3]:font-semibold [&_li]:ml-5 [&_ol]:list-decimal [&_pre]:overflow-auto [&_pre]:rounded-xl [&_pre]:p-3 [&_ul]:list-disc">
                  {markdownContent || '这个 Markdown 文件暂无内容。'}
                </MessageResponse>
              )}
            </div>
          </div>
        ) : previewable ? (
          <iframe src={inlineUrl} title={title} className="bg-background h-full w-full border-0" />
        ) : (
          <div className="grid place-items-center p-6 text-center">
            <div className="grid max-w-sm justify-items-center gap-3">
              <FileTextIcon className="text-muted-foreground size-10" />
              <div>
                <div className="text-sm font-semibold">这个文件类型不能直接预览</div>
                <p className="text-muted-foreground mt-1 text-xs leading-relaxed">
                  可以在新标签页尝试打开，或下载原文件后使用本地应用查看。
                </p>
              </div>
              <div className="flex items-center gap-2">
                <a
                  href={inlineUrl}
                  target="_blank"
                  rel="noreferrer"
                  className="hover:bg-muted inline-flex h-9 items-center gap-2 rounded-full border px-4 text-xs font-semibold"
                >
                  <ExternalLinkIcon className="size-4" />
                  打开
                </a>
                <a
                  href={downloadUrl}
                  className="bg-foreground text-background inline-flex h-9 items-center gap-2 rounded-full px-4 text-xs font-semibold"
                >
                  <DownloadIcon className="size-4" />
                  下载
                </a>
              </div>
            </div>
          </div>
        )}
      </aside>
    </div>
  );
}

export function LiveRagKnowledgePanel({ open, onOpenChange }: LiveRagKnowledgePanelProps) {
  const [mounted, setMounted] = useState(false);
  const [ready, setReady] = useState<LiveRagReady | null>(null);
  const [kbReady, setKbReady] = useState<LiveRagReady | null>(null);
  const [documents, setDocuments] = useState<LiveRagDocumentsResponse | null>(null);
  const [knowledgeBases, setKnowledgeBases] = useState<LiveRagKnowledgeBase[]>([]);
  const [sessionKnowledgeBase, setSessionKnowledgeBaseState] =
    useState<LiveRagSessionKnowledgeBase | null>(null);
  const [selectedKbId, setSelectedKbId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [mutating, setMutating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [newKbName, setNewKbName] = useState('');
  const [newKbDescription, setNewKbDescription] = useState('');
  const [editing, setEditing] = useState(false);
  const [editName, setEditName] = useState('');
  const [editDescription, setEditDescription] = useState('');
  const [pendingDeleteKbId, setPendingDeleteKbId] = useState<string | null>(null);
  const [pendingClear, setPendingClear] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [textImportOpen, setTextImportOpen] = useState(false);
  const [textSource, setTextSource] = useState('manual-note.md');
  const [textContent, setTextContent] = useState('');
  const [previewDocument, setPreviewDocument] = useState<LiveRagDocument | null>(null);

  const selectedKnowledgeBase = useMemo(
    () => knowledgeBases.find((item) => item.kb_id === selectedKbId) ?? null,
    [knowledgeBases, selectedKbId]
  );

  const documentList = documents?.documents ?? [];
  const configuredKbId = sessionKnowledgeBase?.configured?.kb_id;
  const activeKbId = sessionKnowledgeBase?.active_session?.kb_id;
  const locked = sessionKnowledgeBase?.locked === true;
  const normalizedSearchQuery = searchQuery.trim().toLowerCase();

  const visibleDocuments = useMemo(() => {
    if (!normalizedSearchQuery) return documentList;

    return documentList.filter((document) => {
      const haystack = [
        getDocumentTitle(document),
        document.original_filename,
        document.content_summary,
        document.content,
        document.document_id,
        document.file_path,
        document.source_file_path,
      ]
        .filter(Boolean)
        .join(' ')
        .toLowerCase();

      return haystack.includes(normalizedSearchQuery);
    });
  }, [documentList, normalizedSearchQuery]);

  const loadKnowledgeBase = async (kbId: string) => {
    setLoading(true);
    setError(null);

    try {
      const [readyResult, documentResult] = await Promise.all([
        getKnowledgeBaseReady(kbId),
        getKnowledgeBaseDocuments(kbId, 1, 120),
      ]);
      setKbReady(readyResult);
      setDocuments(documentResult);
    } catch (err) {
      setKbReady(null);
      setDocuments(null);
      setError(errorMessage(err, '知识库详情读取失败'));
    } finally {
      setLoading(false);
    }
  };

  const refreshAll = async (preferredKbId?: string) => {
    setLoading(true);
    setError(null);

    try {
      const [readyResult, listResult, sessionResult] = await Promise.all([
        getRagReady(),
        getKnowledgeBases(),
        getSessionKnowledgeBase(),
      ]);
      const nextKnowledgeBases = [...(listResult.knowledge_bases ?? [])].sort((a, b) => {
        if (a.kb_id === 'default') return -1;
        if (b.kb_id === 'default') return 1;
        return a.name.localeCompare(b.name, 'zh-CN');
      });
      const nextSelectedKbId =
        preferredKbId ??
        sessionResult.active_session?.kb_id ??
        sessionResult.configured?.kb_id ??
        selectedKbId ??
        nextKnowledgeBases[0]?.kb_id ??
        null;

      setReady(readyResult);
      setKnowledgeBases(nextKnowledgeBases);
      setSessionKnowledgeBaseState(sessionResult);
      setSelectedKbId(nextSelectedKbId);

      if (nextSelectedKbId) {
        await loadKnowledgeBase(nextSelectedKbId);
      } else {
        setKbReady(null);
        setDocuments(null);
      }
    } catch (err) {
      setError(errorMessage(err, '知识库列表读取失败'));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!open) return;
    if (window.matchMedia('(max-width: 639px)').matches) setViewMode('list');
    void refreshAll();
    setCreating(false);
    setEditing(false);
    setPendingDeleteKbId(null);
    setPendingClear(false);
  }, [open]);

  useEffect(() => {
    if (!open) setPreviewDocument(null);
  }, [open]);

  useEffect(() => {
    if (!selectedKnowledgeBase || editing) return;
    setEditName(selectedKnowledgeBase.name);
    setEditDescription(selectedKnowledgeBase.description ?? '');
  }, [editing, selectedKnowledgeBase]);

  const pollJob = async (kbId: string, jobId?: string) => {
    if (!jobId) return;

    for (let index = 0; index < 6; index += 1) {
      await delay(900);
      const job = await getKnowledgeBaseJob(kbId, jobId);
      const status = job.status?.toLowerCase();
      if (
        job.done ||
        job.finished ||
        status === 'done' ||
        status === 'finished' ||
        status === 'completed' ||
        status === 'processed' ||
        status === 'failed' ||
        status === 'error'
      ) {
        return;
      }
    }
  };

  const handleSelectKnowledgeBase = async (kbId: string) => {
    setPendingDeleteKbId(null);
    setPendingClear(false);
    setPreviewDocument(null);
    setSelectedKbId(kbId);
    setError(null);

    try {
      await loadKnowledgeBase(kbId);
    } catch (err) {
      setError(errorMessage(err, '知识库读取失败'));
    }
  };

  const handleSetDefaultKnowledgeBase = async (kbId: string) => {
    if (kbId === configuredKbId) return;

    setMutating(true);
    setError(null);

    try {
      const sessionResult = await setSessionKnowledgeBase(kbId);
      setSessionKnowledgeBaseState(sessionResult);
      await refreshAll(selectedKbId ?? kbId);
    } catch (err) {
      setError(errorMessage(err, '默认知识库设置失败'));
    } finally {
      setMutating(false);
    }
  };

  const handleCreateKnowledgeBase = async () => {
    const name = newKbName.trim();
    if (!name) return;

    setMutating(true);
    setError(null);

    try {
      const created = await createKnowledgeBase({
        name,
        description: newKbDescription.trim(),
      });
      setNewKbName('');
      setNewKbDescription('');
      setCreating(false);

      if (created.kb_id) {
        await refreshAll(created.kb_id);
      } else {
        await refreshAll();
      }
    } catch (err) {
      setError(errorMessage(err, '知识库创建失败'));
    } finally {
      setMutating(false);
    }
  };

  const handleSaveKnowledgeBase = async () => {
    if (!selectedKnowledgeBase) return;
    const name = editName.trim();
    if (!name) return;

    setMutating(true);
    setError(null);

    try {
      await updateKnowledgeBase(selectedKnowledgeBase.kb_id, {
        name,
        description: editDescription.trim(),
      });
      setEditing(false);
      await refreshAll(selectedKnowledgeBase.kb_id);
    } catch (err) {
      setError(errorMessage(err, '知识库保存失败'));
    } finally {
      setMutating(false);
    }
  };

  const handleDeleteKnowledgeBase = async (knowledgeBase: LiveRagKnowledgeBase) => {
    if (knowledgeBase.kb_id === 'default') return;
    if (pendingDeleteKbId !== knowledgeBase.kb_id) {
      setPendingDeleteKbId(knowledgeBase.kb_id);
      return;
    }

    setMutating(true);
    setError(null);

    try {
      await deleteKnowledgeBase(knowledgeBase.kb_id);
      const fallbackKbId =
        knowledgeBases.find((item) => item.kb_id === 'default')?.kb_id ??
        knowledgeBases.find((item) => item.kb_id !== knowledgeBase.kb_id)?.kb_id;

      if (configuredKbId === knowledgeBase.kb_id && fallbackKbId) {
        const sessionResult = await setSessionKnowledgeBase(fallbackKbId);
        setSessionKnowledgeBaseState(sessionResult);
      }

      await refreshAll(
        selectedKbId === knowledgeBase.kb_id ? fallbackKbId : (selectedKbId ?? fallbackKbId)
      );
      setPendingDeleteKbId(null);
    } catch (err) {
      setError(errorMessage(err, '知识库删除失败'));
    } finally {
      setMutating(false);
    }
  };

  const handleFiles = async (event: ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files?.length || !selectedKbId) return;

    setUploading(true);
    setError(null);

    try {
      const result = await uploadKnowledgeBaseFiles(selectedKbId, files);
      event.target.value = '';
      await pollJob(selectedKbId, result.job_id ?? result.track_id);
      await refreshAll(selectedKbId);
    } catch (err) {
      setError(errorMessage(err, '文件上传失败'));
    } finally {
      setUploading(false);
    }
  };

  const handleTextImport = async () => {
    if (!selectedKbId) return;
    const text = textContent.trim();
    if (!text) return;

    setUploading(true);
    setError(null);

    try {
      const result = await uploadKnowledgeBaseText(selectedKbId, {
        text,
        file_source: textSource.trim() || `manual-note-${Date.now()}.md`,
        document_id: null,
      });
      setTextContent('');
      setTextImportOpen(false);
      await pollJob(selectedKbId, result.job_id ?? result.track_id);
      await refreshAll(selectedKbId);
    } catch (err) {
      setError(errorMessage(err, '文本导入失败'));
    } finally {
      setUploading(false);
    }
  };

  const handleDeleteDocument = async (document: LiveRagDocument) => {
    if (!selectedKbId || !document.document_id) return;

    setError(null);

    try {
      await deleteKnowledgeBaseDocument(selectedKbId, document.document_id);
      if (previewDocument?.document_id === document.document_id) setPreviewDocument(null);
      await refreshAll(selectedKbId);
    } catch (err) {
      setError(errorMessage(err, '文档删除失败'));
    }
  };

  const handleClearDocuments = async () => {
    if (!selectedKbId) return;
    if (!pendingClear) {
      setPendingClear(true);
      return;
    }

    setMutating(true);
    setError(null);

    try {
      await clearKnowledgeBaseDocuments(selectedKbId);
      setPendingClear(false);
      await refreshAll(selectedKbId);
    } catch (err) {
      setError(errorMessage(err, '清空文档失败'));
    } finally {
      setMutating(false);
    }
  };

  if (!mounted) return null;

  return createPortal(
    <div
      className={cn(
        'fixed inset-0 z-[999] flex items-center justify-center p-0 transition-[visibility] sm:p-6',
        open ? 'visible' : 'invisible'
      )}
      aria-hidden={!open}
    >
      <button
        type="button"
        aria-label="关闭医学资料面板"
        className={cn(
          'bg-foreground/8 absolute inset-0 cursor-default backdrop-blur-[2px] transition-opacity duration-200',
          open ? 'opacity-100' : 'pointer-events-none opacity-0'
        )}
        onClick={() => onOpenChange(false)}
      />
      <aside
        role="dialog"
        aria-modal="true"
        aria-label="病历与医学资料管理"
        className={cn(
          'bg-background relative grid h-svh w-full max-w-none grid-cols-1 overflow-hidden border-0 shadow-[0_28px_90px_rgba(0,0,0,0.18)] transition duration-200 ease-out sm:h-[min(690px,calc(100vh-36px))] sm:max-w-[980px] sm:grid-cols-[230px_minmax(0,1fr)] sm:rounded-2xl sm:border',
          open ? 'translate-y-0 scale-100 opacity-100' : 'translate-y-6 scale-[0.98] opacity-0'
        )}
      >
        <section className="bg-muted/40 hidden min-h-0 sm:grid sm:grid-rows-[auto_minmax(0,1fr)_auto]">
          <header className="flex items-center justify-between gap-3 p-3.5">
            <div className="min-w-0">
              <div className="text-muted-foreground text-xs font-semibold tracking-wider">
                MEDAGENT 资料中心
              </div>
              <h2 className="truncate text-base font-semibold">病历与医学资料</h2>
            </div>
            <button
              type="button"
              onClick={() => setCreating((value) => !value)}
              className="hover:bg-background bg-background/75 grid size-8 shrink-0 place-items-center rounded-full"
              aria-label="新建资料库"
            >
              <PlusIcon className="size-4" />
            </button>
          </header>

          <div className="min-h-0 overflow-auto px-2.5 pb-3.5">
            {creating && (
              <div className="bg-background/80 ring-border/45 mb-2 rounded-xl p-2 shadow-sm ring-1">
                <input
                  value={newKbName}
                  onChange={(event) => setNewKbName(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter') void handleCreateKnowledgeBase();
                    if (event.key === 'Escape') setCreating(false);
                  }}
                  autoFocus
                  placeholder="知识库名称"
                  className="bg-background/80 focus:border-foreground h-8 w-full rounded-lg border px-2 text-xs outline-none"
                />
                <input
                  value={newKbDescription}
                  onChange={(event) => setNewKbDescription(event.target.value)}
                  placeholder="描述，可选"
                  className="bg-background/80 focus:border-foreground mt-2 h-8 w-full rounded-lg border px-2 text-xs outline-none"
                />
                <div className="mt-2 flex gap-2">
                  <button
                    type="button"
                    onClick={() => void handleCreateKnowledgeBase()}
                    disabled={mutating || !newKbName.trim()}
                    className="bg-foreground text-background h-8 flex-1 rounded-lg text-xs font-semibold disabled:opacity-50"
                  >
                    创建
                  </button>
                  <button
                    type="button"
                    onClick={() => setCreating(false)}
                    className="bg-muted h-8 flex-1 rounded-lg text-xs font-semibold"
                  >
                    取消
                  </button>
                </div>
              </div>
            )}

            <div className="grid gap-1">
              {knowledgeBases.length === 0 && !loading ? (
                <div className="text-muted-foreground rounded-xl px-3 py-4 text-xs leading-relaxed">
                  暂无知识库，创建后可上传文件并用于下次语音对话。
                </div>
              ) : (
                knowledgeBases.map((knowledgeBase) => (
                  <KnowledgeBaseButton
                    key={knowledgeBase.kb_id}
                    knowledgeBase={knowledgeBase}
                    active={selectedKbId === knowledgeBase.kb_id}
                    configured={configuredKbId === knowledgeBase.kb_id}
                    locked={locked || mutating}
                    pendingDelete={pendingDeleteKbId === knowledgeBase.kb_id}
                    onClick={() => void handleSelectKnowledgeBase(knowledgeBase.kb_id)}
                    onDelete={() => void handleDeleteKnowledgeBase(knowledgeBase)}
                  />
                ))
              )}
            </div>
          </div>

          <footer className="bg-background/80 ring-border/45 m-2.5 rounded-2xl p-3 shadow-sm ring-1">
            <div className="flex items-center justify-between gap-3 text-xs">
              <span className="text-muted-foreground">RAG 服务</span>
              <strong className={ready?.ready ? 'text-emerald-600' : 'text-destructive'}>
                {ready?.ready ? '正常' : '未就绪'}
              </strong>
            </div>
            <div className="text-muted-foreground mt-2 grid gap-1 text-[11px]">
              <div className="flex justify-between gap-3">
                <span>当前锁定</span>
                <strong className="text-foreground max-w-28 truncate">
                  {activeKbId
                    ? (sessionKnowledgeBase?.active_session?.name ?? activeKbId)
                    : locked
                      ? '读取中'
                      : '无'}
                </strong>
              </div>
              <div className="flex justify-between gap-3">
                <span>Embedding</span>
                <strong className="text-foreground max-w-28 truncate">
                  {ready?.embedding_model ?? '未读取'}
                </strong>
              </div>
            </div>
          </footer>
        </section>

        <section className="bg-background grid min-h-0 grid-rows-[auto_minmax(0,1fr)]">
          <header className="border-b p-2.5 sm:p-3">
            <div className="flex items-center justify-between gap-3">
              <div className="min-w-0">
                <div className="flex min-w-0 items-center gap-2">
                  <DatabaseIcon className="size-4 shrink-0" />
                  <h2 className="truncate text-base font-semibold">
                    {selectedKnowledgeBase?.name ?? '病历与医学资料'}
                  </h2>
                  {selectedKbId === activeKbId && (
                    <span className="rounded-full bg-emerald-500/10 px-2 py-0.5 text-[11px] text-emerald-700 dark:text-emerald-300">
                      本次语音使用中
                    </span>
                  )}
                </div>
                <p className="text-muted-foreground mt-1 hidden text-xs sm:block">
                  {selectedKnowledgeBase?.description ||
                    '每组资料独立管理，语音咨询一次使用一组资料。'}
                </p>
              </div>

              <div className="flex shrink-0 items-center gap-2">
                <button
                  type="button"
                  onClick={() => void refreshAll(selectedKbId ?? undefined)}
                  className="hover:bg-muted grid size-8 place-items-center rounded-full border"
                  aria-label="刷新医学资料"
                  disabled={loading}
                >
                  <RefreshCwIcon className={cn('size-4', loading && 'animate-spin')} />
                </button>
                <button
                  type="button"
                  onClick={() => onOpenChange(false)}
                  className="hover:bg-muted grid size-8 place-items-center rounded-full border"
                  aria-label="关闭"
                >
                  <XIcon className="size-4" />
                </button>
              </div>
            </div>

            <div className="mt-2 grid gap-2 sm:hidden">
              <div className="flex gap-2 overflow-x-auto pb-1 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
                {knowledgeBases.map((knowledgeBase) => (
                  <KnowledgeBaseChip
                    key={knowledgeBase.kb_id}
                    knowledgeBase={knowledgeBase}
                    active={selectedKbId === knowledgeBase.kb_id}
                    configured={configuredKbId === knowledgeBase.kb_id}
                    onClick={() => void handleSelectKnowledgeBase(knowledgeBase.kb_id)}
                  />
                ))}
                <button
                  type="button"
                  onClick={() => setCreating((value) => !value)}
                  className="bg-background/75 hover:bg-background inline-flex h-9 shrink-0 items-center gap-2 rounded-full border px-3 text-xs font-semibold"
                >
                  <PlusIcon className="size-3.5" />
                  新建
                </button>
              </div>

              {creating && (
                <div className="bg-muted/35 grid gap-2 rounded-2xl p-3">
                  <input
                    value={newKbName}
                    onChange={(event) => setNewKbName(event.target.value)}
                    placeholder="资料库名称"
                    className="bg-background/80 focus:border-foreground h-9 rounded-xl border px-3 text-xs outline-none"
                  />
                  <input
                    value={newKbDescription}
                    onChange={(event) => setNewKbDescription(event.target.value)}
                    placeholder="描述，可选"
                    className="bg-background/80 focus:border-foreground h-9 rounded-xl border px-3 text-xs outline-none"
                  />
                  <div className="flex justify-end gap-2">
                    <button
                      type="button"
                      onClick={() => setCreating(false)}
                      className="hover:bg-background h-8 rounded-full border px-3 text-xs"
                    >
                      取消
                    </button>
                    <button
                      type="button"
                      onClick={() => void handleCreateKnowledgeBase()}
                      disabled={mutating || !newKbName.trim()}
                      className="bg-foreground text-background h-8 rounded-full px-3 text-xs font-semibold disabled:opacity-50"
                    >
                      创建
                    </button>
                  </div>
                </div>
              )}
            </div>

            {selectedKnowledgeBase && (
              <div className="bg-muted/35 mt-2 grid gap-2 rounded-xl p-2.5">
                {editing ? (
                  <div className="grid gap-2">
                    <div className="grid gap-2 md:grid-cols-[minmax(0,220px)_minmax(0,1fr)]">
                      <input
                        value={editName}
                        onChange={(event) => setEditName(event.target.value)}
                        className="bg-background/80 focus:border-foreground h-9 rounded-xl border px-3 text-xs outline-none"
                      />
                      <input
                        value={editDescription}
                        onChange={(event) => setEditDescription(event.target.value)}
                        placeholder="描述，可选"
                        className="bg-background/80 focus:border-foreground h-9 rounded-xl border px-3 text-xs outline-none"
                      />
                    </div>
                    <div className="flex justify-end gap-2">
                      <button
                        type="button"
                        onClick={() => setEditing(false)}
                        className="hover:bg-background h-8 rounded-full border px-3 text-xs"
                      >
                        取消
                      </button>
                      <button
                        type="button"
                        onClick={() => void handleSaveKnowledgeBase()}
                        disabled={mutating || !editName.trim()}
                        className="bg-foreground text-background inline-flex h-8 items-center gap-1.5 rounded-full px-3 text-xs font-semibold disabled:opacity-50"
                      >
                        <CheckIcon className="size-3.5" />
                        保存
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-wrap items-center justify-between gap-2 sm:gap-3">
                    <div className="text-muted-foreground flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
                      <span>
                        {selectedKnowledgeBase.document_count ?? documents?.total ?? 0} 个文件
                      </span>
                      <span>{selectedKnowledgeBase.chunk_count ?? 0} 个片段</span>
                      <span>状态：{kbReady?.ready ? '已预热' : '未就绪'}</span>
                      {selectedKnowledgeBase.kb_id === 'default' && <span>默认知识库不可删除</span>}
                    </div>
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={() => {
                          if (selectedKbId) void handleSetDefaultKnowledgeBase(selectedKbId);
                        }}
                        disabled={mutating || locked || selectedKbId === configuredKbId}
                        title={
                          locked
                            ? '语音咨询中不能切换默认资料'
                            : selectedKbId === configuredKbId
                              ? '已是默认资料'
                              : '设为默认资料'
                        }
                        className={cn(
                          'inline-flex h-8 items-center gap-1.5 rounded-full border px-3 text-xs transition disabled:cursor-not-allowed',
                          selectedKbId === configuredKbId
                            ? 'bg-muted/70 text-muted-foreground'
                            : 'bg-muted/45 text-muted-foreground hover:bg-background hover:text-foreground',
                          (mutating || locked) && 'opacity-50'
                        )}
                      >
                        <CheckIcon className="size-3.5" />
                        {selectedKbId === configuredKbId ? '当前使用' : '设为默认'}
                      </button>
                      <button
                        type="button"
                        onClick={() => setEditing(true)}
                        className="hover:bg-background inline-flex h-8 items-center gap-1.5 rounded-full border px-3 text-xs"
                      >
                        <PencilIcon className="size-3.5" />
                        编辑
                      </button>
                      <button
                        type="button"
                        onClick={() => void handleClearDocuments()}
                        disabled={mutating || !selectedKbId || documentList.length === 0}
                        className={cn(
                          'inline-flex h-8 items-center gap-1.5 rounded-full border px-3 text-xs disabled:opacity-40',
                          pendingClear
                            ? 'border-destructive/40 bg-destructive/10 text-destructive'
                            : 'hover:bg-background'
                        )}
                      >
                        <Trash2Icon className="size-3.5" />
                        {pendingClear ? '确认清空' : '清空文档'}
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}

            <div className="mt-2 grid gap-2 sm:flex sm:flex-wrap sm:items-center">
              <div className="relative min-w-0 flex-1">
                <SearchIcon className="text-muted-foreground pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2" />
                <input
                  value={searchQuery}
                  onChange={(event) => setSearchQuery(event.target.value)}
                  placeholder="搜索文档"
                  className="border-input bg-muted/35 focus:border-foreground h-9 w-full rounded-full border-transparent pr-3 pl-9 text-xs outline-none"
                />
              </div>

              <div className="flex min-w-0 items-center gap-2 overflow-x-auto [scrollbar-width:none] sm:overflow-visible [&::-webkit-scrollbar]:hidden">
                <button
                  type="button"
                  onClick={() => setTextImportOpen((value) => !value)}
                  disabled={!selectedKbId}
                  className="hover:bg-muted inline-flex h-9 shrink-0 items-center gap-2 rounded-full border px-3.5 text-xs font-semibold disabled:opacity-50"
                >
                  <FilePlus2Icon className="size-4" />
                  文本导入
                </button>

                <label className="bg-foreground text-background hover:bg-foreground/90 inline-flex h-9 shrink-0 cursor-pointer items-center gap-2 rounded-full px-3.5 text-xs font-semibold">
                  <input
                    type="file"
                    multiple
                    className="sr-only"
                    onChange={handleFiles}
                    disabled={!selectedKbId || uploading}
                  />
                  {uploading ? (
                    <Loader2Icon className="size-4 animate-spin" />
                  ) : (
                    <UploadIcon className="size-4" />
                  )}
                  {uploading ? '处理中' : '上传文件'}
                </label>

                <div className="bg-muted/35 inline-flex shrink-0 rounded-full p-1">
                  <button
                    type="button"
                    onClick={() => setViewMode('grid')}
                    className={cn(
                      'grid size-7 place-items-center rounded-full transition',
                      viewMode === 'grid' && 'bg-foreground text-background'
                    )}
                    aria-label="卡片视图"
                    title="卡片视图"
                  >
                    <LayoutGridIcon className="size-4" />
                  </button>
                  <button
                    type="button"
                    onClick={() => setViewMode('list')}
                    className={cn(
                      'grid size-7 place-items-center rounded-full transition',
                      viewMode === 'list' && 'bg-foreground text-background'
                    )}
                    aria-label="列表视图"
                    title="列表视图"
                  >
                    <ListIcon className="size-4" />
                  </button>
                </div>
              </div>
            </div>

            {textImportOpen && (
              <div className="bg-muted/35 mt-3 grid gap-2 rounded-2xl p-3">
                <input
                  value={textSource}
                  onChange={(event) => setTextSource(event.target.value)}
                  placeholder="文件名，例如 门诊记录.md"
                  className="bg-background/80 focus:border-foreground h-9 rounded-xl border px-3 text-xs outline-none"
                />
                <textarea
                  value={textContent}
                  onChange={(event) => setTextContent(event.target.value)}
                  placeholder="粘贴要导入的病历或医学资料"
                  className="bg-background/80 focus:border-foreground min-h-24 resize-y rounded-xl border p-3 text-xs leading-relaxed outline-none"
                />
                <div className="flex justify-end gap-2">
                  <button
                    type="button"
                    onClick={() => setTextImportOpen(false)}
                    className="hover:bg-background h-8 rounded-full border px-3 text-xs"
                  >
                    取消
                  </button>
                  <button
                    type="button"
                    onClick={() => void handleTextImport()}
                    disabled={uploading || !textContent.trim()}
                    className="bg-foreground text-background inline-flex h-8 items-center gap-1.5 rounded-full px-3 text-xs font-semibold disabled:opacity-50"
                  >
                    <UploadIcon className="size-3.5" />
                    导入
                  </button>
                </div>
              </div>
            )}

            {error && (
              <div className="text-destructive mt-3 rounded-xl border p-3 text-sm">{error}</div>
            )}
          </header>

          <div className="bg-muted/35 min-h-0 overflow-auto p-2.5 sm:p-3.5 lg:p-4">
            <div className="mb-2 flex items-center justify-between gap-3 sm:mb-3">
              <h3 className="text-muted-foreground text-xs font-semibold">文档列表</h3>
              <span className="text-muted-foreground text-xs">{visibleDocuments.length} 项</span>
            </div>

            <div className="bg-background/55 ring-border/30 min-h-0 rounded-2xl p-2 shadow-sm ring-1 sm:p-3">
              {loading && documentList.length === 0 ? (
                <div className="grid min-h-72 place-items-center">
                  <Loader2Icon className="text-muted-foreground size-7 animate-spin" />
                </div>
              ) : visibleDocuments.length === 0 ? (
                <div className="bg-background/70 grid min-h-72 place-items-center rounded-2xl">
                  <div className="grid justify-items-center gap-3 text-center">
                    <DatabaseIcon className="text-muted-foreground size-9" />
                    <div>
                      <div className="text-sm font-semibold">这里还没有文件</div>
                      <p className="text-muted-foreground mt-1 text-xs">
                        上传文件或导入文本后，系统会整理资料供咨询时参考。
                      </p>
                    </div>
                    <label className="bg-foreground text-background inline-flex h-9 cursor-pointer items-center gap-2 rounded-full px-4 text-xs font-semibold">
                      <input
                        type="file"
                        multiple
                        className="sr-only"
                        onChange={handleFiles}
                        disabled={!selectedKbId || uploading}
                      />
                      <UploadIcon className="size-4" />
                      上传文件
                    </label>
                  </div>
                </div>
              ) : (
                <div
                  className={cn(
                    viewMode === 'grid'
                      ? 'grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4'
                      : 'grid gap-0.5'
                  )}
                >
                  {visibleDocuments.map((document) => {
                    const canOpenSource = canOpenDocumentSource(document);
                    const sourceUrl =
                      canOpenSource && selectedKbId && document.document_id
                        ? getKnowledgeBaseDocumentSourceUrl(
                            selectedKbId,
                            document.document_id,
                            'inline'
                          )
                        : undefined;

                    return viewMode === 'grid' ? (
                      <DocumentCard
                        key={getDocumentKey(document)}
                        document={document}
                        sourceUrl={sourceUrl}
                        canOpenSource={canOpenSource}
                        onPreview={() => setPreviewDocument(document)}
                        onDelete={() => void handleDeleteDocument(document)}
                      />
                    ) : (
                      <DocumentRow
                        key={getDocumentKey(document)}
                        document={document}
                        canOpenSource={canOpenSource}
                        onPreview={() => setPreviewDocument(document)}
                        onDelete={() => void handleDeleteDocument(document)}
                      />
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        </section>
      </aside>
      <DocumentPreviewPanel
        kbId={selectedKbId}
        document={previewDocument}
        onClose={() => setPreviewDocument(null)}
      />
    </div>,
    document.body
  );
}
