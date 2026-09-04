const DEFAULT_LIVERAG_API_BASE = '/api/liverag';

export const LIVERAG_API_BASE =
  process.env.NEXT_PUBLIC_LIVERAG_API_BASE?.replace(/\/$/, '') ?? DEFAULT_LIVERAG_API_BASE;

export class LiveRagApiError extends Error {
  status?: number;
  type?: string;

  constructor(message: string, status?: number, type?: string) {
    super(message);
    this.name = 'LiveRagApiError';
    this.status = status;
    this.type = type;
  }
}

export interface LiveRagMessage {
  timestamp?: string;
  role: 'user' | 'assistant' | string;
  content: string;
  turn_index?: number | null;
  metadata?: Record<string, unknown>;
}

export interface LiveRagEvidenceChunk {
  chunk_id?: string;
  document_id?: string;
  file_path?: string;
  kb_id?: string;
  kb_name?: string;
  tokens?: number;
  score?: number;
  content?: string;
  content_preview?: string;
}

export interface LiveRagEvidenceDocument {
  document_id?: string;
  file_path?: string;
  title?: string;
  kb_id?: string;
  kb_name?: string;
  chunk_count?: number;
}

export interface LiveRagTurn {
  turn_index: number;
  messages?: LiveRagMessage[];
  user_message?: LiveRagMessage | null;
  assistant_message?: LiveRagMessage | null;
  rag?: {
    status?: 'not_queried' | 'hit' | 'miss' | 'failed' | string;
    queried?: boolean;
    hit?: boolean;
    has_context?: boolean;
    kb_id?: string;
    kb_name?: string;
    query?: string;
    effective_query?: string;
    request_id?: string;
    latency_ms?: number;
    cache_hit?: boolean;
    evidence_documents?: LiveRagEvidenceDocument[];
    evidence_chunks?: LiveRagEvidenceChunk[];
    evidence_count?: number;
    no_evidence_reason?: string | null;
    error?: { message?: string } | string | null;
    context_preview?: string;
  };
  rag_contexts?: unknown[];
}

export interface LiveRagRagContext {
  turn_index?: number | null;
  [key: string]: unknown;
}

export interface LiveRagPage<T> {
  items: T[];
  next_cursor?: number | null;
  has_more?: boolean;
  total?: number;
  order?: string;
}

export interface LiveRagKnowledgeBaseRef {
  kb_id: string;
  name: string;
}

export interface LiveRagActiveKnowledgeBase extends LiveRagKnowledgeBaseRef {
  locked_at?: string;
  job_id?: string;
  room_id?: string;
}

export interface LiveRagSessionKnowledgeBase {
  configured?: LiveRagKnowledgeBaseRef | null;
  active_session?: LiveRagActiveKnowledgeBase | null;
  locked?: boolean;
  pending_reconnect?: boolean;
}

export interface LiveRagRuntimeState {
  active_session?: {
    started_at?: string;
    job_id?: string;
    room_id?: string;
    room?: string;
    voice?: LiveRagModelConfig;
    knowledge_base?: LiveRagActiveKnowledgeBase;
  } | null;
  turn_index?: number;
  last_user_text?: string;
  previous_user_text?: string;
  last_rag_query?: string;
  rag_tool_mode?: 'auto' | 'never' | string;
  last_assistant_chars?: number;
  last_tts_text_chars?: number;
  last_answer_too_long?: boolean;
  last_answer_used_rag?: boolean;
  active_voice_model?: LiveRagModelConfig;
  knowledge_base?: LiveRagSessionKnowledgeBase;
  last_rag?: {
    hit?: boolean;
    has_context?: boolean;
    request_id?: string;
    metrics?: Record<string, unknown>;
    error?: unknown;
  };
}

export interface LiveRagReady {
  ready?: boolean;
  initialized?: boolean;
  provider_configured?: boolean;
  llm_model?: string;
  embedding_model?: string;
  embedding_dim?: number;
  knowledge_bases_dir?: string;
  cached_kb_ids?: string[];
  working_dir?: string;
  user_data_dir?: string;
  upload_dir?: string;
  workspace?: string;
}

export interface LiveRagKnowledgeBase {
  kb_id: string;
  name: string;
  description?: string;
  document_count?: number;
  chunk_count?: number;
  created_at?: string;
  updated_at?: string;
}

export interface LiveRagKnowledgeBaseListResponse {
  knowledge_bases?: LiveRagKnowledgeBase[];
  total?: number;
}

export interface LiveRagOverview {
  summary?: {
    total_documents?: number;
    processed_documents?: number;
    failed_documents?: number;
    pending_documents?: number;
    total_chunks?: number;
    total_entities?: number;
    total_relationships?: number;
  };
  topics?: unknown[];
  top_entities?: unknown[];
  top_relationships?: unknown[];
  documents?: LiveRagDocument[];
}

export interface LiveRagKnowledgeBaseContextOverview {
  kb_id?: string;
  content?: string;
  meta?: {
    kb_id?: string;
    updated_at?: string;
    stale?: boolean;
    reason?: string;
    source?: string;
    source_job_id?: string;
  };
}

export interface LiveRagDocument {
  document_id?: string;
  kb_id?: string;
  kb_name?: string;
  original_filename?: string;
  status?: string;
  parse_status?: 'pending' | 'parsed' | 'failed' | string;
  index_status?: 'pending' | 'processing' | 'processed' | 'failed' | string;
  content_summary?: string;
  content_length?: number;
  content?: string;
  chunks?: unknown[];
  file_path?: string;
  source_file_path?: string;
  source_file_exists?: boolean;
  source_file_size?: number;
  source_sha256?: string;
  content_type?: string;
  extension?: string;
  track_id?: string;
  chunks_count?: number;
  created_at?: string;
  updated_at?: string;
  error_msg?: string | null;
  raw?: Record<string, unknown>;
  status_raw?: Record<string, unknown>;
}

export interface LiveRagDocumentsResponse {
  documents?: LiveRagDocument[];
  total?: number;
  page?: number;
  page_size?: number;
  total_pages?: number;
  has_next?: boolean;
  has_prev?: boolean;
  status_counts?: Record<string, number>;
}

export interface LiveRagDocumentUploadResult {
  track_id?: string;
  job_id?: string;
  processing_mode?: string;
  kb_id?: string;
  kb_name?: string;
  document?: LiveRagDocument;
  parsed_count?: number;
  error_count?: number;
  total_files?: number;
  files?: Array<{
    filename?: string;
    extension?: string;
    text_len?: number;
    kb_id?: string;
    kb_name?: string;
  }>;
  errors?: unknown[];
}

export interface LiveRagJob {
  job_id?: string;
  kb_id?: string;
  track_id?: string;
  status?: string;
  total_files?: number;
  parsed_count?: number;
  failed_count?: number;
  error_msg?: string | null;
  created_at?: string;
  updated_at?: string;
  documents?: LiveRagDocument[];
  total?: number;
  done?: boolean;
  finished?: boolean;
  error?: unknown;
  progress?: number;
  raw?: unknown;
}

export interface LiveRagConfig {
  enabled?: boolean;
  base_url?: string;
  context_path?: string;
  overview_path?: string;
  api_key?: string;
  api_key_masked?: string;
  api_key_set?: boolean;
  rag_tool_mode?: 'auto' | 'never' | string;
  query_mode?: 'local' | 'global' | 'hybrid' | 'naive' | 'mix' | 'bypass' | string;
  timeout_ms?: number;
  top_k?: number;
  chunk_top_k?: number;
  context_max_chars?: number;
  cache_ttl_s?: number;
  enable_rerank?: boolean;
}

export interface LiveRagModelOptionValue {
  id: string;
  label?: string;
  verified?: boolean;
}

export interface LiveRagModelConfigField {
  key: string;
  label?: string;
  type?: 'text' | 'secret' | 'url' | string;
  required?: boolean;
  description?: string;
  default?: string | number | boolean;
  default_value?: string | number | boolean;
}

export interface LiveRagVoiceProviderOption {
  provider: string;
  label?: string;
  description?: string;
  models?: LiveRagModelOptionValue[];
  voices?: LiveRagModelOptionValue[];
  default_model?: string;
  default_voice?: string;
  config_fields?: LiveRagModelConfigField[];
}

export interface LiveRagModelOptions {
  stt?: {
    providers?: LiveRagVoiceProviderOption[];
    default_provider?: string;
  };
  llm?: {
    mode?: string;
    description?: string;
    config_fields?: LiveRagModelConfigField[];
  };
  tts?: {
    providers?: LiveRagVoiceProviderOption[];
    default_provider?: string;
  };
}

export interface LiveRagVoiceLlmConfig {
  model?: string;
  base_url?: string;
  api_key?: string;
  api_key_set?: boolean;
  api_key_masked?: string;
  effective?: string;
  [key: string]: string | boolean | number | undefined;
}

export interface LiveRagVoiceSttConfig {
  provider?: string;
  model?: string;
  app_id?: string;
  app_id_set?: boolean;
  app_id_masked?: string;
  access_token?: string;
  access_token_set?: boolean;
  access_token_masked?: string;
  effective?: string;
  [key: string]: string | boolean | number | undefined;
}

export interface LiveRagVoiceTtsConfig {
  provider?: string;
  model?: string;
  voice?: string;
  api_key?: string;
  api_key_set?: boolean;
  api_key_masked?: string;
  effective?: string;
  [key: string]: string | boolean | number | undefined;
}

export interface LiveRagModelConfig {
  voice?: {
    llm?: LiveRagVoiceLlmConfig;
    stt?: LiveRagVoiceSttConfig;
    tts?: LiveRagVoiceTtsConfig;
  };
  options?: LiveRagModelOptions;
}

export interface LiveRagModelEffectiveState {
  configured?: LiveRagModelConfig;
  active_session?: LiveRagModelConfig | null;
  pending_reconnect?: boolean;
}

export interface LiveRagContextModelConfig {
  model?: string;
  base_url?: string;
  api_key?: string;
  api_key_set?: boolean;
  api_key_masked?: string;
  temperature?: number;
  max_tokens?: number;
  max_session_chars?: number;
  history_reference_limit?: number;
  timeout_ms?: number;
  effective?: string;
}

export interface LiveRagQueryRequest {
  query: string;
  profile?: string;
  mode?: string;
  top_k?: number;
  chunk_top_k?: number;
  context_max_chars?: number;
  include_references?: boolean;
  include_chunk_content?: boolean;
  last_query?: string;
  rewrite_followup?: boolean;
}

export interface LiveRagQueryContextResponse {
  hit?: boolean;
  query?: string;
  effective_query?: string;
  rewritten?: boolean;
  context?: string;
  context_truncated?: boolean;
  references?: LiveRagEvidenceDocument[];
  chunks?: LiveRagEvidenceChunk[];
}

export type LiveRagQueryDataResponse = Record<string, unknown>;

interface LiveRagContextModelResponse {
  context_model: LiveRagContextModelConfig;
}

interface Envelope<T> {
  request_id?: string;
  status: 'ok' | 'error' | string;
  data: T;
  metrics?: Record<string, unknown>;
  error?: { message?: string; type?: string } | null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

async function parseError(response: Response) {
  try {
    const payload = (await response.json()) as unknown;
    if (isRecord(payload)) {
      const detail = payload.detail;
      const envelopeError = isRecord(payload.error) ? payload.error : undefined;
      const message =
        (typeof detail === 'string' && detail) ||
        (typeof envelopeError?.message === 'string' && envelopeError.message) ||
        (typeof payload.message === 'string' && payload.message) ||
        response.statusText;
      const type =
        (typeof envelopeError?.type === 'string' && envelopeError.type) ||
        (typeof detail === 'string' && detail) ||
        undefined;
      return new LiveRagApiError(message, response.status, type);
    }
  } catch {
    // Keep the original HTTP status when the backend does not return JSON.
  }

  return new LiveRagApiError(
    `医学资料服务 ${response.status}: ${response.statusText}`,
    response.status
  );
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const headers =
    init?.body instanceof FormData
      ? init.headers
      : {
          'Content-Type': 'application/json',
          ...(init?.headers as Record<string, string> | undefined),
        };

  const requestUrl = path.startsWith('/api/') ? path : `${LIVERAG_API_BASE}${path}`;
  const response = await fetch(requestUrl, {
    ...init,
    headers,
  });

  if (!response.ok) {
    throw await parseError(response);
  }

  return (await response.json()) as T;
}

async function fetchEnvelope<T>(path: string, init?: RequestInit): Promise<T> {
  const envelope = await fetchJson<Envelope<T>>(path, init);

  if (envelope.status !== 'ok') {
    throw new LiveRagApiError(
      envelope.error?.message ?? '医学资料服务暂时不可用',
      undefined,
      envelope.error?.type
    );
  }

  return envelope.data;
}

function encodePath(value: string) {
  return encodeURIComponent(value);
}

export function getRuntimeState() {
  return fetchJson<LiveRagRuntimeState>('/runtime/state');
}

export function getSessionTurns(limit = 8) {
  return fetchJson<LiveRagTurn[]>(`/session/turns?limit=${limit}`);
}

export function getCurrentVoiceSessionTurns(limit = 50) {
  return fetchEnvelope<LiveRagPage<LiveRagTurn>>(`/api/voice/current/turns?limit=${limit}`);
}

export function getCurrentVoiceSessionRagContext(limit = 50) {
  return fetchEnvelope<LiveRagPage<LiveRagRagContext>>(
    `/api/voice/current/rag-context?limit=${limit}`
  );
}

export async function getCurrentVoiceSessionEvidence(limit = 50) {
  const [turnPage, ragPage] = await Promise.all([
    getCurrentVoiceSessionTurns(limit),
    getCurrentVoiceSessionRagContext(limit),
  ]);
  const contextsByTurn = new Map<number, LiveRagRagContext[]>();
  for (const context of ragPage.items) {
    if (typeof context.turn_index !== 'number') continue;
    const contexts = contextsByTurn.get(context.turn_index) ?? [];
    contexts.push(context);
    contextsByTurn.set(context.turn_index, contexts);
  }
  return turnPage.items.map((turn) => ({
    ...turn,
    rag_contexts: contextsByTurn.get(turn.turn_index) ?? turn.rag_contexts ?? [],
  }));
}

export function clearSession() {
  return fetchJson<{ status?: string }>('/session/clear', { method: 'POST' });
}

export function getSessionKnowledgeBase() {
  return fetchJson<LiveRagSessionKnowledgeBase>('/session/knowledge-base');
}

export function setSessionKnowledgeBase(kbId: string) {
  return fetchJson<LiveRagSessionKnowledgeBase>('/session/knowledge-base', {
    method: 'PUT',
    body: JSON.stringify({ kb_id: kbId }),
  });
}

export function getSoulPrompt() {
  return fetchJson<{ content?: string }>('/prompt/soul');
}

export function updateSoulPrompt(content: string) {
  return fetchJson<{ status?: string }>('/prompt/soul', {
    method: 'PUT',
    body: JSON.stringify({ content }),
  });
}

export function getRagReady() {
  return fetchEnvelope<LiveRagReady>('/rag/ready');
}

export function getRagConfig() {
  return fetchEnvelope<LiveRagConfig>('/rag/config');
}

export function updateRagConfig(config: Partial<LiveRagConfig>) {
  return fetchEnvelope<LiveRagConfig>('/rag/config', {
    method: 'PUT',
    body: JSON.stringify(config),
  });
}

export function getKnowledgeBases() {
  return fetchEnvelope<LiveRagKnowledgeBaseListResponse>('/rag/knowledge-bases');
}

export function createKnowledgeBase(input: { name: string; description?: string }) {
  return fetchEnvelope<LiveRagKnowledgeBase>('/rag/knowledge-bases', {
    method: 'POST',
    body: JSON.stringify(input),
  });
}

export function getKnowledgeBase(kbId: string) {
  return fetchEnvelope<LiveRagKnowledgeBase>(`/rag/knowledge-bases/${encodePath(kbId)}`);
}

export function updateKnowledgeBase(kbId: string, input: { name?: string; description?: string }) {
  return fetchEnvelope<LiveRagKnowledgeBase>(`/rag/knowledge-bases/${encodePath(kbId)}`, {
    method: 'PATCH',
    body: JSON.stringify(input),
  });
}

export function deleteKnowledgeBase(kbId: string) {
  return fetchEnvelope<unknown>(`/rag/knowledge-bases/${encodePath(kbId)}`, {
    method: 'DELETE',
  });
}

export function getKnowledgeBaseReady(kbId: string) {
  return fetchEnvelope<LiveRagReady>(`/rag/knowledge-bases/${encodePath(kbId)}/ready`);
}

export function getKnowledgeBaseContextOverview(kbId: string) {
  return fetchEnvelope<LiveRagKnowledgeBaseContextOverview>(
    `/rag/knowledge-bases/${encodePath(kbId)}/context/overview`
  );
}

export function updateKnowledgeBaseContextOverview(kbId: string, content: string) {
  return fetchEnvelope<LiveRagKnowledgeBaseContextOverview>(
    `/rag/knowledge-bases/${encodePath(kbId)}/context/overview`,
    {
      method: 'PUT',
      body: JSON.stringify({ content }),
    }
  );
}

export function getKnowledgeBaseDocuments(kbId: string, page = 1, pageSize = 80) {
  return fetchEnvelope<LiveRagDocumentsResponse>(
    `/rag/knowledge-bases/${encodePath(kbId)}/documents?page=${page}&page_size=${pageSize}`
  );
}

export function getKnowledgeBaseDocument(kbId: string, documentId: string) {
  return fetchEnvelope<LiveRagDocument>(
    `/rag/knowledge-bases/${encodePath(kbId)}/documents/${encodePath(documentId)}`
  );
}

export function getKnowledgeBaseDocumentSourceUrl(
  kbId: string,
  documentId: string,
  disposition: 'inline' | 'attachment' = 'inline'
) {
  return `${LIVERAG_API_BASE}/rag/knowledge-bases/${encodePath(kbId)}/documents/${encodePath(
    documentId
  )}/source?disposition=${disposition}`;
}

export function uploadKnowledgeBaseText(
  kbId: string,
  input: { text: string; file_source: string; document_id?: string | null }
) {
  return fetchEnvelope<LiveRagDocumentUploadResult>(
    `/rag/knowledge-bases/${encodePath(kbId)}/documents/text`,
    {
      method: 'POST',
      body: JSON.stringify(input),
    }
  );
}

export function uploadKnowledgeBaseFiles(kbId: string, files: FileList | File[]) {
  const formData = new FormData();
  Array.from(files).forEach((file) => formData.append('files', file));

  return fetchEnvelope<LiveRagDocumentUploadResult>(
    `/rag/knowledge-bases/${encodePath(kbId)}/documents/files`,
    {
      method: 'POST',
      body: formData,
    }
  );
}

export function getKnowledgeBaseJob(kbId: string, jobId: string) {
  return fetchEnvelope<LiveRagJob>(
    `/rag/knowledge-bases/${encodePath(kbId)}/jobs/${encodePath(jobId)}`
  );
}

export function deleteKnowledgeBaseDocument(kbId: string, documentId: string) {
  return fetchEnvelope<unknown>(
    `/rag/knowledge-bases/${encodePath(kbId)}/documents/${encodePath(documentId)}`,
    {
      method: 'DELETE',
    }
  );
}

export function clearKnowledgeBaseDocuments(kbId: string) {
  return fetchEnvelope<unknown>(`/rag/knowledge-bases/${encodePath(kbId)}/documents`, {
    method: 'DELETE',
  });
}

export function getModelConfig() {
  return fetchEnvelope<LiveRagModelConfig>('/model/config');
}

export function getModelOptions() {
  return fetchEnvelope<LiveRagModelOptions>('/model/options');
}

export function updateModelConfig(config: LiveRagModelConfig) {
  return fetchEnvelope<LiveRagModelConfig>('/model/config', {
    method: 'PUT',
    body: JSON.stringify(config),
  });
}

export function getModelEffectiveState() {
  return fetchEnvelope<LiveRagModelEffectiveState>('/model/effective-state');
}

export async function getContextModelConfig() {
  const data = await fetchEnvelope<LiveRagContextModelResponse>('/model/context-config');
  return data.context_model;
}

export async function updateContextModelConfig(config: Partial<LiveRagContextModelConfig>) {
  const data = await fetchEnvelope<LiveRagContextModelResponse>('/model/context-config', {
    method: 'PUT',
    body: JSON.stringify(config),
  });
  return data.context_model;
}

export function queryKnowledgeBaseContext(kbId: string, request: LiveRagQueryRequest) {
  return fetchEnvelope<LiveRagQueryContextResponse>(
    `/rag/knowledge-bases/${encodePath(kbId)}/query/context`,
    {
      method: 'POST',
      body: JSON.stringify(request),
    }
  );
}

export function queryKnowledgeBaseData(kbId: string, request: LiveRagQueryRequest) {
  return fetchEnvelope<LiveRagQueryDataResponse>(
    `/rag/knowledge-bases/${encodePath(kbId)}/query/data`,
    {
      method: 'POST',
      body: JSON.stringify(request),
    }
  );
}

export function querySessionContext(request: LiveRagQueryRequest) {
  return fetchEnvelope<LiveRagQueryContextResponse>('/rag/session-query/context', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

export function querySessionData(request: LiveRagQueryRequest) {
  return fetchEnvelope<LiveRagQueryDataResponse>('/rag/session-query/data', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}
