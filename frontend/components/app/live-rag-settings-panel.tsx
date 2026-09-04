'use client';

import { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import { useTheme } from 'next-themes';
import {
  ChevronDownIcon,
  EyeIcon,
  EyeOffIcon,
  Loader2Icon,
  SaveIcon,
  SettingsIcon,
  XIcon,
} from 'lucide-react';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  type LiveRagConfig,
  type LiveRagContextModelConfig,
  type LiveRagKnowledgeBase,
  type LiveRagKnowledgeBaseContextOverview,
  type LiveRagModelConfig,
  type LiveRagModelConfigField,
  type LiveRagModelEffectiveState,
  type LiveRagModelOptions,
  type LiveRagVoiceProviderOption,
  getContextModelConfig,
  getKnowledgeBaseContextOverview,
  getKnowledgeBases,
  getModelConfig,
  getModelEffectiveState,
  getModelOptions,
  getRagConfig,
  getSessionKnowledgeBase,
  getSoulPrompt,
  updateContextModelConfig,
  updateKnowledgeBaseContextOverview,
  updateModelConfig,
  updateRagConfig,
  updateSoulPrompt,
} from '@/lib/liverag-api';
import { cn } from '@/lib/shadcn/utils';

interface LiveRagSettingsPanelProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

type SettingsTab = 'general' | 'retrieval' | 'voice' | 'context' | 'agent';

const SETTINGS_TABS: Array<{ value: SettingsTab; label: string; description: string }> = [
  { value: 'general', label: '通用', description: '显示与资料引用' },
  { value: 'retrieval', label: '资料检索', description: '医学资料查询参数' },
  { value: 'voice', label: '语音模型', description: '高级语音服务配置' },
  { value: 'context', label: '资料上下文', description: '概览与历史整理' },
  { value: 'agent', label: '助手设定', description: '高级行为配置' },
];

function NumberInput({
  label,
  value,
  min,
  max,
  onChange,
}: {
  label: string;
  value: number;
  min?: number;
  max?: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="grid gap-1.5">
      <span className="text-muted-foreground text-[11px] font-medium">{label}</span>
      <input
        type="number"
        min={min}
        max={max}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="border-input bg-background/80 focus:border-foreground h-8 rounded-lg border px-2.5 text-xs outline-none"
      />
    </label>
  );
}

function TextInput({
  label,
  value,
  onChange,
  placeholder,
  type = 'text',
  hint,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  type?: 'text' | 'password';
  hint?: string;
}) {
  return (
    <label className="grid gap-1.5">
      <span className="text-muted-foreground text-[11px] font-medium">{label}</span>
      <input
        type={type}
        value={value}
        placeholder={placeholder}
        onChange={(event) => onChange(event.target.value)}
        className="border-input bg-background/80 focus:border-foreground h-8 rounded-lg border px-2.5 text-xs outline-none"
      />
      {hint && <span className="text-muted-foreground text-[10px]">{hint}</span>}
    </label>
  );
}

function SecretInput({
  label,
  value,
  onChange,
  placeholder,
  maskedValue,
  configured,
  description,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  maskedValue?: string;
  configured?: boolean;
  description?: string;
}) {
  const [visible, setVisible] = useState(false);
  const stateHint = value
    ? '已从后端回填，可直接编辑'
    : maskedValue
      ? `当前：${maskedValue}`
      : configured
        ? '已配置，后端未返回原文'
        : '未配置';
  const hint = [stateHint, description].filter(Boolean).join(' · ');

  return (
    <label className="grid gap-1.5">
      <span className="text-muted-foreground text-[11px] font-medium">{label}</span>
      <span className="relative block">
        <input
          type={visible ? 'text' : 'password'}
          value={value}
          placeholder={placeholder ?? maskedValue ?? '未配置'}
          onChange={(event) => onChange(event.target.value)}
          className="border-input bg-background/80 focus:border-foreground h-8 w-full rounded-lg border px-2.5 pr-9 text-xs outline-none"
        />
        <button
          type="button"
          onClick={() => setVisible((current) => !current)}
          className="text-muted-foreground hover:text-foreground absolute top-1/2 right-1 grid size-6 -translate-y-1/2 place-items-center rounded-md"
          aria-label={visible ? '隐藏密钥' : '显示密钥'}
          title={visible ? '隐藏密钥' : '显示密钥'}
        >
          {visible ? <EyeOffIcon className="size-3.5" /> : <EyeIcon className="size-3.5" />}
        </button>
      </span>
      <span className="text-muted-foreground text-[10px]">{hint}</span>
    </label>
  );
}

function SettingsSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="grid gap-2">
      <h3 className="text-muted-foreground px-1 text-sm font-semibold">{title}</h3>
      <div className="bg-muted/40 overflow-hidden rounded-2xl px-3">{children}</div>
    </section>
  );
}

function ModelConfigGroup({
  title,
  description,
  badge,
  children,
}: {
  title: string;
  description: string;
  badge: string;
  children: React.ReactNode;
}) {
  return (
    <section className="bg-background/70 ring-border/45 grid gap-3 rounded-2xl p-3 ring-1">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h4 className="text-sm font-semibold">{title}</h4>
          <p className="text-muted-foreground mt-0.5 text-[11px] leading-relaxed">{description}</p>
        </div>
        <span className="bg-muted text-muted-foreground shrink-0 rounded-full px-2 py-1 text-[10px] font-semibold">
          {badge}
        </span>
      </div>
      <div className="grid gap-2.5 sm:grid-cols-2">{children}</div>
    </section>
  );
}

function SettingsTabButton({
  item,
  active,
  onClick,
}: {
  item: (typeof SETTINGS_TABS)[number];
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'rounded-xl px-2.5 py-2 text-center transition md:px-3 md:text-left',
        active
          ? 'bg-foreground text-background shadow-sm'
          : 'text-foreground hover:bg-background/75'
      )}
    >
      <span className="block truncate text-xs font-semibold sm:text-sm">{item.label}</span>
      <span
        className={cn(
          'mt-0.5 hidden truncate text-[11px] md:block',
          active ? 'text-background/70' : 'text-muted-foreground'
        )}
      >
        {item.description}
      </span>
    </button>
  );
}

function MobileSettingsTabPicker({
  activeTab,
  expanded,
  onToggle,
  onChange,
}: {
  activeTab: SettingsTab;
  expanded: boolean;
  onToggle: () => void;
  onChange: (value: SettingsTab) => void;
}) {
  const activeItem = SETTINGS_TABS.find((item) => item.value === activeTab) ?? SETTINGS_TABS[0];

  return (
    <nav className="bg-muted/40 grid gap-1 rounded-2xl p-1.5 sm:hidden">
      <button
        type="button"
        onClick={onToggle}
        className="bg-background/80 flex h-14 items-center justify-between gap-3 rounded-xl px-3 text-left"
        aria-expanded={expanded}
      >
        <span className="min-w-0">
          <span className="block truncate text-sm font-semibold">{activeItem.label}</span>
          <span className="text-muted-foreground mt-0.5 block truncate text-[11px]">
            {activeItem.description}
          </span>
        </span>
        <ChevronDownIcon className={cn('size-4 shrink-0 transition', expanded && 'rotate-180')} />
      </button>

      {expanded && (
        <div className="grid gap-1 border-t pt-1">
          {SETTINGS_TABS.map((item) => (
            <button
              key={item.value}
              type="button"
              onClick={() => onChange(item.value)}
              className={cn(
                'rounded-xl px-3 py-2 text-left transition',
                activeTab === item.value
                  ? 'bg-foreground text-background'
                  : 'hover:bg-background/75'
              )}
            >
              <span className="block truncate text-sm font-semibold">{item.label}</span>
              <span
                className={cn(
                  'mt-0.5 block truncate text-[11px]',
                  activeTab === item.value ? 'text-background/70' : 'text-muted-foreground'
                )}
              >
                {item.description}
              </span>
            </button>
          ))}
        </div>
      )}
    </nav>
  );
}

function SettingsRow({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="border-border/55 grid min-h-[54px] items-center gap-2.5 border-b py-2.5 last:border-b-0 md:grid-cols-[minmax(140px,1fr)_minmax(200px,320px)]">
      <div className="min-w-0">
        <div className="text-sm font-semibold">{label}</div>
        {hint && <div className="text-muted-foreground mt-0.5 text-[11px]">{hint}</div>}
      </div>
      <div className="min-w-0 justify-self-stretch md:justify-self-end">{children}</div>
    </div>
  );
}

function SelectBox({
  value,
  onChange,
  options,
}: {
  value: string;
  onChange: (value: string) => void;
  options: Array<{ value: string; label: string }>;
}) {
  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger className="bg-background/75 hover:bg-background h-9 w-full rounded-xl border-transparent px-3 text-xs font-semibold shadow-none">
        <SelectValue />
      </SelectTrigger>
      <SelectContent className="z-[1100] rounded-xl border-0 p-1 shadow-[0_16px_42px_rgba(0,0,0,0.14)]">
        {options.map((option) => (
          <SelectItem key={option.value} value={option.value} className="rounded-lg text-xs">
            {option.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

function OptionSelect({
  value,
  onChange,
  options,
}: {
  value: string;
  onChange: (value: string) => void;
  options: Array<{ value: string; label: string }>;
}) {
  if (options.length === 0) {
    return (
      <div className="bg-background/60 text-muted-foreground grid h-9 place-items-center rounded-xl px-3 text-xs">
        暂无可用选项
      </div>
    );
  }

  return <SelectBox value={value || options[0].value} onChange={onChange} options={options} />;
}

function getProviderOptions(options: LiveRagModelOptions | null, section: 'stt' | 'tts') {
  return options?.[section]?.providers ?? [];
}

function getDefaultProviderId(options: LiveRagModelOptions | null, section: 'stt' | 'tts') {
  const providers = getProviderOptions(options, section);
  return options?.[section]?.default_provider ?? providers[0]?.provider ?? '';
}

function findProvider(
  options: LiveRagModelOptions | null,
  section: 'stt' | 'tts',
  providerId?: string
) {
  const providers = getProviderOptions(options, section);
  return providers.find((provider) => provider.provider === providerId) ?? providers[0] ?? null;
}

function providerLabel(provider: LiveRagVoiceProviderOption) {
  return provider.label ?? provider.provider;
}

function optionLabel(option: { id: string; label?: string; verified?: boolean }) {
  return option.label ?? option.id;
}

function configString(
  config: Record<string, string | boolean | number | undefined> | undefined,
  key: string
) {
  const value = config?.[key];
  return typeof value === 'string' ? value : '';
}

function defaultFieldValue(field: LiveRagModelConfigField) {
  const value = field.default_value ?? field.default;
  if (value === undefined) return undefined;
  return String(value);
}

function maskedFieldValue(
  config: Record<string, string | boolean | number | undefined> | undefined,
  key: string
) {
  return configString(config, `${key}_masked`);
}

function configuredFieldValue(
  config: Record<string, string | boolean | number | undefined> | undefined,
  key: string
) {
  return config?.[`${key}_set`] === true;
}

function fieldPatch(fields: LiveRagModelConfigField[]) {
  return Object.fromEntries(
    fields
      .map((field) => [field.key, defaultFieldValue(field)])
      .filter(([, value]) => value !== undefined)
  ) as Record<string, string>;
}

function modelId(provider: LiveRagVoiceProviderOption | null, current?: string) {
  const models = provider?.models ?? [];
  if (current && models.some((model) => model.id === current)) return current;
  return provider?.default_model ?? models[0]?.id ?? '';
}

function voiceId(provider: LiveRagVoiceProviderOption | null, current?: string) {
  const voices = provider?.voices ?? [];
  if (current && voices.some((voice) => voice.id === current)) return current;
  return provider?.default_voice ?? voices[0]?.id ?? '';
}

function normalizeVoiceConfig(config: LiveRagModelConfig, options: LiveRagModelOptions) {
  const sttProviderId = config.voice?.stt?.provider ?? getDefaultProviderId(options, 'stt');
  const ttsProviderId = config.voice?.tts?.provider ?? getDefaultProviderId(options, 'tts');
  const sttProvider = findProvider(options, 'stt', sttProviderId);
  const ttsProvider = findProvider(options, 'tts', ttsProviderId);

  return {
    ...config,
    voice: {
      ...config.voice,
      stt: {
        ...config.voice?.stt,
        provider: sttProvider?.provider ?? sttProviderId,
        model: modelId(sttProvider, config.voice?.stt?.model),
      },
      tts: {
        ...config.voice?.tts,
        provider: ttsProvider?.provider ?? ttsProviderId,
        model: modelId(ttsProvider, config.voice?.tts?.model),
        voice: voiceId(ttsProvider, config.voice?.tts?.voice),
      },
    },
  } satisfies LiveRagModelConfig;
}

function ProviderConfigFields({
  fields,
  config,
  onChange,
}: {
  fields: LiveRagModelConfigField[];
  config: Record<string, string | boolean | number | undefined> | undefined;
  onChange: (key: string, value: string) => void;
}) {
  if (fields.length === 0) {
    return (
      <div className="text-muted-foreground rounded-xl border border-dashed p-3 text-xs sm:col-span-2">
        这个 provider 没有额外配置项。
      </div>
    );
  }

  return (
    <>
      {fields.map((field) =>
        field.type === 'secret' ? (
          <SecretInput
            key={field.key}
            label={`${field.label ?? field.key}${field.required ? ' *' : ''}`}
            value={configString(config, field.key)}
            placeholder={maskedFieldValue(config, field.key) || field.label || field.key}
            maskedValue={maskedFieldValue(config, field.key)}
            configured={configuredFieldValue(config, field.key)}
            description={field.description}
            onChange={(value) => onChange(field.key, value)}
          />
        ) : (
          <TextInput
            key={field.key}
            label={`${field.label ?? field.key}${field.required ? ' *' : ''}`}
            value={configString(config, field.key)}
            placeholder={field.label ?? field.key}
            type={field.type === 'url' ? 'text' : 'text'}
            hint={field.description}
            onChange={(value) => onChange(field.key, value)}
          />
        )
      )}
    </>
  );
}

function ToggleSwitch({
  checked,
  onChange,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className={cn(
        'ml-auto flex h-6 w-11 items-center rounded-full p-1 transition',
        checked ? 'bg-foreground' : 'bg-foreground/20'
      )}
    >
      <span
        className={cn(
          'bg-background size-4 rounded-full shadow-sm transition',
          checked && 'translate-x-5'
        )}
      />
    </button>
  );
}

function ThemeSettingControl() {
  const { theme, resolvedTheme, setTheme } = useTheme();
  const currentTheme = theme ?? 'system';

  const applyTheme = (nextTheme: string) => {
    const normalizedTheme = nextTheme as 'light' | 'dark' | 'system';
    setTheme(normalizedTheme);

    const root = document.documentElement;
    const systemTheme = window.matchMedia('(prefers-color-scheme: dark)').matches
      ? 'dark'
      : 'light';
    const effectiveTheme = normalizedTheme === 'system' ? systemTheme : normalizedTheme;

    root.classList.toggle('dark', effectiveTheme === 'dark');
    root.style.colorScheme = effectiveTheme;
    window.localStorage.setItem('theme', normalizedTheme);
  };

  return (
    <SelectBox
      value={currentTheme}
      onChange={applyTheme}
      options={[
        {
          value: 'system',
          label: `跟随系统${resolvedTheme ? ` · 当前${resolvedTheme === 'dark' ? '深色' : '浅色'}` : ''}`,
        },
        { value: 'light', label: '浅色模式' },
        { value: 'dark', label: '深色模式' },
      ]}
    />
  );
}

function stringOrUndefined(value?: string) {
  const normalized = value?.trim();
  return normalized ? normalized : undefined;
}

function numberOrUndefined(value?: number) {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function compactRecord<T extends Record<string, unknown>>(input: T) {
  return Object.fromEntries(
    Object.entries(input).filter(([, value]) => value !== undefined)
  ) as Partial<T>;
}

function buildProviderPayload(
  config: Record<string, string | boolean | number | undefined> | undefined,
  provider: LiveRagVoiceProviderOption | null,
  extra: Record<string, string | undefined>
) {
  const fieldValues = Object.fromEntries(
    (provider?.config_fields ?? []).map((field) => [
      field.key,
      stringOrUndefined(configString(config, field.key)),
    ])
  );

  return compactRecord({
    provider: stringOrUndefined(provider?.provider ?? configString(config, 'provider')),
    ...extra,
    ...fieldValues,
  });
}

function buildModelConfigPayload(
  config: LiveRagModelConfig,
  options: LiveRagModelOptions
): LiveRagModelConfig {
  const llm = config.voice?.llm;
  const stt = config.voice?.stt;
  const tts = config.voice?.tts;
  const sttProvider = findProvider(options, 'stt', stt?.provider);
  const ttsProvider = findProvider(options, 'tts', tts?.provider);
  const voice: NonNullable<LiveRagModelConfig['voice']> = {};
  const llmPayload = compactRecord({
    model: stringOrUndefined(llm?.model),
    base_url: stringOrUndefined(llm?.base_url),
    api_key: stringOrUndefined(llm?.api_key),
  });
  const sttPayload = buildProviderPayload(stt, sttProvider, {
    model: stringOrUndefined(stt?.model),
  });
  const ttsPayload = buildProviderPayload(tts, ttsProvider, {
    model: stringOrUndefined(tts?.model),
    voice: stringOrUndefined(tts?.voice),
  });

  if (Object.keys(llmPayload).length > 0) voice.llm = llmPayload;
  if (Object.keys(sttPayload).length > 0) voice.stt = sttPayload;
  if (Object.keys(ttsPayload).length > 0) voice.tts = ttsPayload;

  return Object.keys(voice).length > 0 ? { voice } : {};
}

function buildContextModelConfigPayload(
  config: LiveRagContextModelConfig
): Partial<LiveRagContextModelConfig> {
  return compactRecord({
    model: stringOrUndefined(config.model),
    base_url: stringOrUndefined(config.base_url),
    api_key: stringOrUndefined(config.api_key),
    temperature: numberOrUndefined(config.temperature),
    max_tokens: numberOrUndefined(config.max_tokens),
    max_session_chars: numberOrUndefined(config.max_session_chars),
    history_reference_limit: numberOrUndefined(config.history_reference_limit),
    timeout_ms: numberOrUndefined(config.timeout_ms),
  });
}

export function LiveRagSettingsPanel({ open, onOpenChange }: LiveRagSettingsPanelProps) {
  const [mounted, setMounted] = useState(false);
  const [activeTab, setActiveTab] = useState<SettingsTab>('general');
  const [tabMenuOpen, setTabMenuOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [modelError, setModelError] = useState<string | null>(null);
  const [contextModelError, setContextModelError] = useState<string | null>(null);
  const [contextOverviewError, setContextOverviewError] = useState<string | null>(null);
  const [modelConfigAvailable, setModelConfigAvailable] = useState(false);
  const [contextModelConfigAvailable, setContextModelConfigAvailable] = useState(false);
  const [contextOverviewAvailable, setContextOverviewAvailable] = useState(false);
  const [config, setConfig] = useState<LiveRagConfig>({});
  const [modelConfig, setModelConfig] = useState<LiveRagModelConfig>({});
  const [modelOptions, setModelOptions] = useState<LiveRagModelOptions | null>(null);
  const [contextModelConfig, setContextModelConfig] = useState<LiveRagContextModelConfig>({});
  const [knowledgeBases, setKnowledgeBases] = useState<LiveRagKnowledgeBase[]>([]);
  const [contextKbId, setContextKbId] = useState<string | null>(null);
  const [contextOverview, setContextOverview] =
    useState<LiveRagKnowledgeBaseContextOverview | null>(null);
  const [contextOverviewContent, setContextOverviewContent] = useState('');
  const [modelEffectiveState, setModelEffectiveState] = useState<LiveRagModelEffectiveState | null>(
    null
  );
  const [soulPrompt, setSoulPrompt] = useState('');

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!open) return;
    setTabMenuOpen(false);

    let cancelled = false;

    const loadSettings = async () => {
      setLoading(true);
      setError(null);
      setModelError(null);
      setContextModelError(null);
      setContextOverviewError(null);
      setSaved(false);

      try {
        const [ragConfig, soulResult] = await Promise.all([getRagConfig(), getSoulPrompt()]);
        const [
          modelConfigResult,
          modelOptionsResult,
          modelEffectiveResult,
          contextModelConfigResult,
          knowledgeBasesResult,
          sessionKnowledgeBaseResult,
        ] = await Promise.allSettled([
          getModelConfig(),
          getModelOptions(),
          getModelEffectiveState(),
          getContextModelConfig(),
          getKnowledgeBases(),
          getSessionKnowledgeBase(),
        ]);

        if (cancelled) return;
        setConfig(ragConfig);
        setSoulPrompt(soulResult.content ?? '');

        if (modelConfigResult.status === 'fulfilled' && modelOptionsResult.status === 'fulfilled') {
          setModelOptions(modelOptionsResult.value);
          setModelConfig(normalizeVoiceConfig(modelConfigResult.value, modelOptionsResult.value));
          setModelConfigAvailable(true);
        } else {
          setModelConfig({});
          setModelOptions(null);
          setModelConfigAvailable(false);
          setModelError('语音模型配置或选项接口暂不可用，请确认后端已按新版 API 启动。');
        }

        if (modelEffectiveResult.status === 'fulfilled') {
          setModelEffectiveState(modelEffectiveResult.value);
        } else {
          setModelEffectiveState(null);
        }

        if (contextModelConfigResult.status === 'fulfilled') {
          setContextModelConfig(contextModelConfigResult.value);
          setContextModelConfigAvailable(true);
        } else {
          setContextModelConfig({});
          setContextModelConfigAvailable(false);
          setContextModelError('知识库上下文模型接口暂不可用，请确认后端已按新版 API 启动。');
        }

        if (knowledgeBasesResult.status === 'fulfilled') {
          const nextKnowledgeBases = [...(knowledgeBasesResult.value.knowledge_bases ?? [])].sort(
            (a, b) => {
              if (a.kb_id === 'default') return -1;
              if (b.kb_id === 'default') return 1;
              return a.name.localeCompare(b.name, 'zh-CN');
            }
          );
          const sessionKnowledgeBase =
            sessionKnowledgeBaseResult.status === 'fulfilled'
              ? sessionKnowledgeBaseResult.value
              : null;
          const nextContextKbId =
            sessionKnowledgeBase?.active_session?.kb_id ??
            sessionKnowledgeBase?.configured?.kb_id ??
            contextKbId ??
            nextKnowledgeBases[0]?.kb_id ??
            null;

          setKnowledgeBases(nextKnowledgeBases);
          setContextKbId(nextContextKbId);

          if (nextContextKbId) {
            try {
              const overviewResult = await getKnowledgeBaseContextOverview(nextContextKbId);
              if (cancelled) return;
              setContextOverview(overviewResult);
              setContextOverviewContent(overviewResult.content ?? '');
              setContextOverviewAvailable(true);
            } catch (err) {
              if (cancelled) return;
              setContextOverview(null);
              setContextOverviewContent('');
              setContextOverviewAvailable(false);
              setContextOverviewError(err instanceof Error ? err.message : '知识库概览读取失败');
            }
          } else {
            setContextOverview(null);
            setContextOverviewContent('');
            setContextOverviewAvailable(false);
          }
        } else {
          setKnowledgeBases([]);
          setContextKbId(null);
          setContextOverview(null);
          setContextOverviewContent('');
          setContextOverviewAvailable(false);
          setContextOverviewError('知识库列表读取失败，请确认后端已启动。');
        }
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : '设置读取失败');
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    void loadSettings();

    return () => {
      cancelled = true;
    };
  }, [open]);

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    setSaved(false);

    try {
      const saveTasks: Array<Promise<unknown>> = [
        updateRagConfig({
          enabled: config.enabled ?? true,
          rag_tool_mode: config.rag_tool_mode === 'never' ? 'never' : 'auto',
          query_mode: config.query_mode ?? 'naive',
          timeout_ms: config.timeout_ms ?? 900,
          top_k: config.top_k ?? 4,
          chunk_top_k: config.chunk_top_k ?? 4,
          context_max_chars: config.context_max_chars ?? 1800,
          cache_ttl_s: config.cache_ttl_s ?? 45,
          enable_rerank: config.enable_rerank ?? false,
        }).then((nextConfig) => {
          setConfig(nextConfig);
        }),
        updateSoulPrompt(soulPrompt),
      ];

      if (modelConfigAvailable && modelOptions) {
        saveTasks.push(
          updateModelConfig(buildModelConfigPayload(modelConfig, modelOptions)).then(
            (nextModelConfig) => {
              setModelConfig(normalizeVoiceConfig(nextModelConfig, modelOptions));
            }
          )
        );
      }

      if (contextModelConfigAvailable) {
        saveTasks.push(
          updateContextModelConfig(buildContextModelConfigPayload(contextModelConfig)).then(
            (nextContextModelConfig) => {
              setContextModelConfig(nextContextModelConfig);
            }
          )
        );
      }

      if (contextKbId && contextOverviewAvailable) {
        saveTasks.push(
          updateKnowledgeBaseContextOverview(contextKbId, contextOverviewContent).then(
            (nextOverview) => {
              setContextOverview(nextOverview);
              setContextOverviewContent(nextOverview.content ?? '');
            }
          )
        );
      }

      await Promise.all(saveTasks);
      if (modelConfigAvailable && modelOptions) {
        try {
          setModelEffectiveState(await getModelEffectiveState());
        } catch {
          setModelEffectiveState(null);
        }
      }
      setSaved(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : '设置保存失败');
    } finally {
      setSaving(false);
    }
  };

  const updateVoiceModel = (section: 'llm' | 'stt' | 'tts', patch: Record<string, string>) => {
    setModelConfig((current) => ({
      ...current,
      voice: {
        ...current.voice,
        [section]: {
          ...current.voice?.[section],
          ...patch,
        },
      },
    }));
  };

  const updateVoiceProvider = (section: 'stt' | 'tts', providerId: string) => {
    if (!modelOptions) return;

    const provider = findProvider(modelOptions, section, providerId);
    if (!provider) return;

    setModelConfig((current) => ({
      ...current,
      voice: {
        ...current.voice,
        [section]: {
          provider: provider.provider,
          model: modelId(provider),
          ...(section === 'tts' ? { voice: voiceId(provider) } : {}),
          ...fieldPatch(provider.config_fields ?? []),
        },
      },
    }));
  };

  const updateContextModel = (patch: Partial<LiveRagContextModelConfig>) => {
    setContextModelConfig((current) => ({
      ...current,
      ...patch,
    }));
  };

  const handleContextKnowledgeBaseChange = async (kbId: string) => {
    setContextKbId(kbId);
    setContextOverviewError(null);
    setContextOverview(null);
    setContextOverviewContent('');
    setContextOverviewAvailable(false);

    try {
      const overviewResult = await getKnowledgeBaseContextOverview(kbId);
      setContextOverview(overviewResult);
      setContextOverviewContent(overviewResult.content ?? '');
      setContextOverviewAvailable(true);
    } catch (err) {
      setContextOverviewError(err instanceof Error ? err.message : '知识库概览读取失败');
    }
  };

  const sttProvider = findProvider(modelOptions, 'stt', modelConfig.voice?.stt?.provider);
  const ttsProvider = findProvider(modelOptions, 'tts', modelConfig.voice?.tts?.provider);
  const sttProviderOptions = getProviderOptions(modelOptions, 'stt').map((provider) => ({
    value: provider.provider,
    label: providerLabel(provider),
  }));
  const ttsProviderOptions = getProviderOptions(modelOptions, 'tts').map((provider) => ({
    value: provider.provider,
    label: providerLabel(provider),
  }));
  const sttModelOptions =
    sttProvider?.models?.map((model) => ({
      value: model.id,
      label: optionLabel(model),
    })) ?? [];
  const ttsModelOptions =
    ttsProvider?.models?.map((model) => ({
      value: model.id,
      label: optionLabel(model),
    })) ?? [];
  const ttsVoiceOptions =
    ttsProvider?.voices?.map((voice) => ({
      value: voice.id,
      label: optionLabel(voice),
    })) ?? [];

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
        aria-label="关闭设置面板"
        className={cn(
          'bg-foreground/8 absolute inset-0 cursor-default backdrop-blur-[2px] transition-opacity duration-200',
          open ? 'opacity-100' : 'pointer-events-none opacity-0'
        )}
        onClick={() => onOpenChange(false)}
      />
      <aside
        role="dialog"
        aria-modal="true"
        aria-label="设置"
        className={cn(
          'bg-background relative grid h-svh w-full max-w-none grid-rows-[auto_minmax(0,1fr)_auto] overflow-hidden border-0 shadow-[0_28px_90px_rgba(0,0,0,0.18)] transition duration-200 ease-out sm:h-[min(700px,calc(100vh-36px))] sm:max-w-[980px] sm:rounded-2xl sm:border',
          open ? 'translate-y-0 scale-100 opacity-100' : 'translate-y-6 scale-[0.98] opacity-0'
        )}
      >
        <header className="flex items-start justify-between gap-4 border-b px-4 py-3 sm:px-5 sm:py-3.5">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <SettingsIcon className="size-4" />
              <h2 className="truncate text-base font-semibold">设置</h2>
            </div>
            <p className="text-muted-foreground mt-1 text-[11px]">
              显示、医学资料检索、语音与助手高级配置。
            </p>
          </div>
          <button
            type="button"
            onClick={() => onOpenChange(false)}
            className="hover:bg-muted grid size-8 shrink-0 place-items-center rounded-full border sm:size-9"
            aria-label="关闭"
          >
            <XIcon className="size-4" />
          </button>
        </header>

        <div className="min-h-0 overflow-auto p-3 sm:p-3.5 md:p-4">
          {loading ? (
            <div className="grid h-full min-h-80 place-items-center">
              <Loader2Icon className="text-muted-foreground size-7 animate-spin" />
            </div>
          ) : (
            <div className="mx-auto grid max-w-[920px] gap-3 md:h-full md:grid-cols-[170px_minmax(0,1fr)]">
              <MobileSettingsTabPicker
                activeTab={activeTab}
                expanded={tabMenuOpen}
                onToggle={() => setTabMenuOpen((current) => !current)}
                onChange={(value) => {
                  setActiveTab(value);
                  setTabMenuOpen(false);
                }}
              />
              <nav className="bg-muted/40 hidden grid-cols-5 gap-1 self-start rounded-2xl p-1.5 sm:grid md:auto-rows-min md:grid-cols-1 md:p-2">
                {SETTINGS_TABS.map((item) => (
                  <SettingsTabButton
                    key={item.value}
                    item={item}
                    active={activeTab === item.value}
                    onClick={() => setActiveTab(item.value)}
                  />
                ))}
              </nav>

              <div className="min-w-0">
                {activeTab === 'general' && (
                  <div className="grid gap-3 sm:gap-4">
                    <SettingsSection title="通用设置">
                      <SettingsRow label="界面显示">
                        <ThemeSettingControl />
                      </SettingsRow>
                      <SettingsRow label="语音引用医学资料" hint="关闭后语音咨询不会查询资料">
                        <ToggleSwitch
                          checked={config.enabled ?? true}
                          onChange={(checked) =>
                            setConfig((current) => ({ ...current, enabled: checked }))
                          }
                        />
                      </SettingsRow>
                    </SettingsSection>
                  </div>
                )}

                {activeTab === 'retrieval' && (
                  <div className="grid gap-3 sm:gap-4">
                    <SettingsSection title="医学资料检索">
                      <SettingsRow label="资料调用方式">
                        <SelectBox
                          value={config.rag_tool_mode === 'never' ? 'never' : 'auto'}
                          onChange={(value) =>
                            setConfig((current) => ({ ...current, rag_tool_mode: value }))
                          }
                          options={[
                            { value: 'auto', label: '自动 · 模型决定' },
                            { value: 'never', label: '从不 · 禁用查询' },
                          ]}
                        />
                      </SettingsRow>
                      <SettingsRow label="检索模式">
                        <SelectBox
                          value={config.query_mode ?? 'naive'}
                          onChange={(value) =>
                            setConfig((current) => ({ ...current, query_mode: value }))
                          }
                          options={[
                            { value: 'naive', label: 'naive' },
                            { value: 'mix', label: 'mix' },
                            { value: 'local', label: 'local' },
                            { value: 'global', label: 'global' },
                            { value: 'hybrid', label: 'hybrid' },
                            { value: 'bypass', label: 'bypass' },
                          ]}
                        />
                      </SettingsRow>
                      <SettingsRow label="Rerank" hint="启用后可能提高质量，但会增加延迟">
                        <ToggleSwitch
                          checked={config.enable_rerank ?? false}
                          onChange={(checked) =>
                            setConfig((current) => ({ ...current, enable_rerank: checked }))
                          }
                        />
                      </SettingsRow>
                    </SettingsSection>

                    <SettingsSection title="查询参数">
                      <div className="grid gap-2.5 py-3 sm:grid-cols-2 lg:grid-cols-4">
                        <NumberInput
                          label="top_k"
                          min={1}
                          max={80}
                          value={config.top_k ?? 4}
                          onChange={(value) =>
                            setConfig((current) => ({ ...current, top_k: value }))
                          }
                        />
                        <NumberInput
                          label="chunk_top_k"
                          min={1}
                          max={40}
                          value={config.chunk_top_k ?? 4}
                          onChange={(value) =>
                            setConfig((current) => ({ ...current, chunk_top_k: value }))
                          }
                        />
                        <NumberInput
                          label="timeout_ms"
                          min={100}
                          max={10000}
                          value={config.timeout_ms ?? 900}
                          onChange={(value) =>
                            setConfig((current) => ({ ...current, timeout_ms: value }))
                          }
                        />
                        <NumberInput
                          label="上下文字符"
                          min={200}
                          max={12000}
                          value={config.context_max_chars ?? 1800}
                          onChange={(value) =>
                            setConfig((current) => ({ ...current, context_max_chars: value }))
                          }
                        />
                      </div>
                    </SettingsSection>
                  </div>
                )}

                {activeTab === 'voice' && (
                  <SettingsSection title="语音模型配置">
                    <div className="grid gap-3 py-3">
                      {modelError && (
                        <div className="text-muted-foreground rounded-xl border border-dashed p-3 text-xs">
                          {modelError}
                        </div>
                      )}
                      {modelEffectiveState?.pending_reconnect && (
                        <div className="rounded-xl border border-amber-500/25 bg-amber-500/10 p-3 text-xs text-amber-700 dark:text-amber-300">
                          新的语音模型配置会在挂断并重新开始通话后生效。
                        </div>
                      )}
                      <div className="grid gap-3">
                        <ModelConfigGroup
                          title="对话大模型"
                          description="负责理解问题、生成回复和调用知识库工具。"
                          badge="LLM"
                        >
                          <TextInput
                            label="模型"
                            value={modelConfig.voice?.llm?.model ?? ''}
                            placeholder="gemma-4-e4b-it-4bit"
                            onChange={(value) => updateVoiceModel('llm', { model: value })}
                          />
                          <TextInput
                            label="Base URL"
                            value={modelConfig.voice?.llm?.base_url ?? ''}
                            placeholder="http://127.0.0.1:8000/v1"
                            onChange={(value) => updateVoiceModel('llm', { base_url: value })}
                          />
                          <div className="sm:col-span-2">
                            <SecretInput
                              label="API Key"
                              value={modelConfig.voice?.llm?.api_key ?? ''}
                              placeholder={modelConfig.voice?.llm?.api_key_masked ?? 'LLM API Key'}
                              maskedValue={modelConfig.voice?.llm?.api_key_masked}
                              configured={modelConfig.voice?.llm?.api_key_set}
                              onChange={(value) => updateVoiceModel('llm', { api_key: value })}
                            />
                          </div>
                        </ModelConfigGroup>

                        <ModelConfigGroup
                          title="语音识别"
                          description={sttProvider?.description ?? '负责把用户实时语音转成文本。'}
                          badge={sttProvider ? providerLabel(sttProvider) : 'STT'}
                        >
                          <label className="grid gap-1.5">
                            <span className="text-muted-foreground text-[11px] font-medium">
                              Provider
                            </span>
                            <OptionSelect
                              value={modelConfig.voice?.stt?.provider ?? ''}
                              onChange={(value) => updateVoiceProvider('stt', value)}
                              options={sttProviderOptions}
                            />
                          </label>
                          <label className="grid gap-1.5">
                            <span className="text-muted-foreground text-[11px] font-medium">
                              模型
                            </span>
                            <OptionSelect
                              value={modelConfig.voice?.stt?.model ?? ''}
                              onChange={(value) => updateVoiceModel('stt', { model: value })}
                              options={sttModelOptions}
                            />
                          </label>
                          <ProviderConfigFields
                            fields={sttProvider?.config_fields ?? []}
                            config={modelConfig.voice?.stt}
                            onChange={(key, value) => updateVoiceModel('stt', { [key]: value })}
                          />
                        </ModelConfigGroup>

                        <ModelConfigGroup
                          title="语音合成"
                          description={ttsProvider?.description ?? '负责把助手回复转换成实时语音。'}
                          badge={ttsProvider ? providerLabel(ttsProvider) : 'TTS'}
                        >
                          <label className="grid gap-1.5">
                            <span className="text-muted-foreground text-[11px] font-medium">
                              Provider
                            </span>
                            <OptionSelect
                              value={modelConfig.voice?.tts?.provider ?? ''}
                              onChange={(value) => updateVoiceProvider('tts', value)}
                              options={ttsProviderOptions}
                            />
                          </label>
                          <label className="grid gap-1.5">
                            <span className="text-muted-foreground text-[11px] font-medium">
                              模型
                            </span>
                            <OptionSelect
                              value={modelConfig.voice?.tts?.model ?? ''}
                              onChange={(value) => updateVoiceModel('tts', { model: value })}
                              options={ttsModelOptions}
                            />
                          </label>
                          <label className="grid gap-1.5">
                            <span className="text-muted-foreground text-[11px] font-medium">
                              音色
                            </span>
                            <OptionSelect
                              value={modelConfig.voice?.tts?.voice ?? ''}
                              onChange={(value) => updateVoiceModel('tts', { voice: value })}
                              options={ttsVoiceOptions}
                            />
                          </label>
                          <ProviderConfigFields
                            fields={ttsProvider?.config_fields ?? []}
                            config={modelConfig.voice?.tts}
                            onChange={(key, value) => updateVoiceModel('tts', { [key]: value })}
                          />
                        </ModelConfigGroup>
                      </div>
                    </div>
                  </SettingsSection>
                )}

                {activeTab === 'context' && (
                  <div className="grid gap-3 sm:gap-4">
                    <SettingsSection title="资料上下文模型">
                      <div className="grid gap-3 py-3">
                        {contextModelError && (
                          <div className="text-muted-foreground rounded-xl border border-dashed p-3 text-xs">
                            {contextModelError}
                          </div>
                        )}
                        <ModelConfigGroup
                          title="资料整理模型"
                          description="用于生成资料概览，并在语音结束后整理本次咨询历史。"
                          badge={contextModelConfig.effective ?? 'Context'}
                        >
                          <TextInput
                            label="模型"
                            value={contextModelConfig.model ?? ''}
                            placeholder="qwen-max"
                            onChange={(value) => updateContextModel({ model: value })}
                          />
                          <TextInput
                            label="Base URL"
                            value={contextModelConfig.base_url ?? ''}
                            placeholder="https://dashscope.aliyuncs.com/compatible-mode/v1"
                            onChange={(value) => updateContextModel({ base_url: value })}
                          />
                          <div className="sm:col-span-2">
                            <SecretInput
                              label="API Key"
                              value={contextModelConfig.api_key ?? ''}
                              placeholder={contextModelConfig.api_key_masked ?? 'Context API Key'}
                              maskedValue={contextModelConfig.api_key_masked}
                              configured={contextModelConfig.api_key_set}
                              onChange={(value) => updateContextModel({ api_key: value })}
                            />
                          </div>
                          <NumberInput
                            label="temperature"
                            min={0}
                            max={2}
                            value={contextModelConfig.temperature ?? 0}
                            onChange={(value) => updateContextModel({ temperature: value })}
                          />
                          <NumberInput
                            label="max_tokens"
                            min={256}
                            max={32000}
                            value={contextModelConfig.max_tokens ?? 2000}
                            onChange={(value) => updateContextModel({ max_tokens: value })}
                          />
                          <NumberInput
                            label="会话压缩字符"
                            min={1000}
                            max={100000}
                            value={contextModelConfig.max_session_chars ?? 16000}
                            onChange={(value) => updateContextModel({ max_session_chars: value })}
                          />
                          <NumberInput
                            label="历史引用条数"
                            min={0}
                            max={50}
                            value={contextModelConfig.history_reference_limit ?? 8}
                            onChange={(value) =>
                              updateContextModel({ history_reference_limit: value })
                            }
                          />
                          <NumberInput
                            label="timeout_ms"
                            min={1000}
                            max={120000}
                            value={contextModelConfig.timeout_ms ?? 15000}
                            onChange={(value) => updateContextModel({ timeout_ms: value })}
                          />
                        </ModelConfigGroup>
                      </div>
                    </SettingsSection>

                    <SettingsSection title="资料概览">
                      <div className="grid gap-3 py-3">
                        {knowledgeBases.length > 0 ? (
                          <SettingsRow label="当前资料库" hint="每组资料维护独立概览">
                            <SelectBox
                              value={contextKbId ?? knowledgeBases[0]?.kb_id ?? ''}
                              onChange={(value) => void handleContextKnowledgeBaseChange(value)}
                              options={knowledgeBases.map((knowledgeBase) => ({
                                value: knowledgeBase.kb_id,
                                label: knowledgeBase.name,
                              }))}
                            />
                          </SettingsRow>
                        ) : (
                          <div className="text-muted-foreground rounded-xl border border-dashed p-3 text-xs">
                            暂无可用资料库。
                          </div>
                        )}

                        {contextOverviewError && (
                          <div className="text-muted-foreground rounded-xl border border-dashed p-3 text-xs">
                            {contextOverviewError}
                          </div>
                        )}

                        <label className="grid gap-2">
                          <span className="text-sm font-semibold">资料概览</span>
                          <textarea
                            value={contextOverviewContent}
                            onChange={(event) => setContextOverviewContent(event.target.value)}
                            disabled={!contextKbId || !contextOverviewAvailable}
                            placeholder="当前资料库的概览"
                            className="border-input bg-background/80 focus:border-foreground min-h-[180px] resize-y rounded-xl border p-3 text-xs leading-relaxed outline-none disabled:opacity-60"
                          />
                        </label>

                        <div className="text-muted-foreground flex flex-wrap gap-x-3 gap-y-1 text-[11px]">
                          <span>
                            状态：
                            {contextOverview?.meta?.stale
                              ? '待重新生成'
                              : contextOverviewAvailable
                                ? '可编辑'
                                : '未读取'}
                          </span>
                          <span>来源：{contextOverview?.meta?.source ?? '未记录'}</span>
                          <span>更新：{contextOverview?.meta?.updated_at ?? '未记录'}</span>
                        </div>
                      </div>
                    </SettingsSection>
                  </div>
                )}

                {activeTab === 'agent' && (
                  <SettingsSection title="助手设定">
                    <div className="grid gap-3 py-3">
                      <label className="grid gap-2">
                        <span className="text-sm font-semibold">SOUL.md</span>
                        <span className="text-muted-foreground text-[11px]">
                          此处用于高级助手行为配置；系统安全规则由服务端维护。
                        </span>
                        <textarea
                          value={soulPrompt}
                          onChange={(event) => setSoulPrompt(event.target.value)}
                          className="border-input bg-background/80 focus:border-foreground min-h-[220px] resize-y rounded-xl border p-3 text-xs leading-relaxed outline-none"
                        />
                      </label>
                    </div>
                  </SettingsSection>
                )}
              </div>
            </div>
          )}

          {error && (
            <div className="text-destructive mt-4 rounded-xl border p-3 text-sm">{error}</div>
          )}
          {saved && !error && (
            <div className="mt-4 rounded-xl border border-emerald-500/25 bg-emerald-500/10 p-3 text-sm text-emerald-700 dark:text-emerald-300">
              设置已保存。
            </div>
          )}
        </div>

        <footer className="flex items-center justify-end gap-2 border-t p-3 sm:p-3.5">
          <button
            type="button"
            onClick={() => onOpenChange(false)}
            className="hover:bg-muted h-9 rounded-full border px-4 text-xs font-medium"
          >
            取消
          </button>
          <button
            type="button"
            onClick={handleSave}
            disabled={saving || loading}
            className="bg-foreground text-background hover:bg-foreground/90 inline-flex h-9 items-center gap-2 rounded-full px-4 text-xs font-semibold disabled:opacity-60"
          >
            {saving ? (
              <Loader2Icon className="size-4 animate-spin" />
            ) : (
              <SaveIcon className="size-4" />
            )}
            保存设置
          </button>
        </footer>
      </aside>
    </div>,
    document.body
  );
}
