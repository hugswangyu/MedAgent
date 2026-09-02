'use client';

import { type ComponentProps } from 'react';
import { AnimatePresence } from 'motion/react';
import { type AgentState, type ReceivedMessage } from '@livekit/components-react';
import { AgentChatIndicator } from '@/components/agents-ui/agent-chat-indicator';
import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from '@/components/ai-elements/conversation';
import { Message, MessageContent, MessageResponse } from '@/components/ai-elements/message';
import type { LiveRagEvidenceChunk, LiveRagTurn } from '@/lib/liverag-api';
import { decodeLiveRagDisplayText, getLiveRagDisplayName } from '@/lib/liverag-display';

function findTurnForMessage(message: string, turns: LiveRagTurn[], assistantIndex: number) {
  const exactTurn = turns.find((turn) => turn.assistant_message?.content === message);

  if (exactTurn) return exactTurn;

  const assistantTurns = turns.filter((turn) => turn.assistant_message?.content);
  return assistantTurns[assistantIndex];
}

function formatScore(score?: number) {
  if (typeof score !== 'number') return null;
  return score.toFixed(2);
}

function evidenceLabel(chunk: LiveRagEvidenceChunk) {
  const filePath = chunk.file_path
    ? getLiveRagDisplayName(chunk.file_path)
    : decodeLiveRagDisplayText(chunk.document_id) || '知识库片段';
  const score = formatScore(chunk.score);
  return score ? `${filePath} · ${score}` : filePath;
}

function RagEvidence({ turn }: { turn?: LiveRagTurn }) {
  const rag = turn?.rag;
  const chunks = rag?.evidence_chunks ?? [];
  const documents = rag?.evidence_documents ?? [];
  const hasEvidence = chunks.length > 0 || documents.length > 0;

  if (!rag || rag.status === 'not_queried') return null;

  return (
    <div className="mt-3 grid gap-2 border-t pt-3">
      <div className="flex items-center justify-between gap-3 text-xs font-semibold">
        <span>本次回答依据</span>
        <span className="text-muted-foreground">
          {rag.status === 'hit'
            ? `${rag.evidence_count ?? chunks.length} 个来源${rag.latency_ms ? ` · ${Math.round(rag.latency_ms)}ms` : ''}`
            : rag.status === 'miss'
              ? '未命中'
              : '查询失败'}
        </span>
      </div>
      {hasEvidence && (
        <div className="flex flex-wrap gap-2">
          {chunks.slice(0, 4).map((chunk, index) => (
            <span
              key={chunk.chunk_id ?? `${chunk.document_id}-${index}`}
              title={chunk.content_preview}
              className="bg-muted text-muted-foreground inline-flex max-w-full rounded-full border px-2 py-1 text-xs"
            >
              {evidenceLabel(chunk)}
            </span>
          ))}
          {chunks.length === 0 &&
            documents.slice(0, 4).map((document) => (
              <span
                key={document.document_id ?? document.file_path}
                className="bg-muted text-muted-foreground inline-flex max-w-full rounded-full border px-2 py-1 text-xs"
              >
                {document.file_path
                  ? getLiveRagDisplayName(document.file_path)
                  : decodeLiveRagDisplayText(document.title ?? document.document_id) ||
                    '知识库文档'}
              </span>
            ))}
        </div>
      )}
      {!hasEvidence && rag.context_preview && (
        <p className="text-muted-foreground line-clamp-2 text-xs">{rag.context_preview}</p>
      )}
      {rag.no_evidence_reason && (
        <p className="text-muted-foreground text-xs">无依据原因：{rag.no_evidence_reason}</p>
      )}
    </div>
  );
}

/**
 * Props for the AgentChatTranscript component.
 */
export interface AgentChatTranscriptProps extends ComponentProps<'div'> {
  /**
   * The current state of the agent. When 'thinking', displays a loading indicator.
   */
  agentState?: AgentState;
  /**
   * Array of messages to display in the transcript.
   * @defaultValue []
   */
  messages?: ReceivedMessage[];
  /**
   * LiveRAG turn data used to append "本次回答依据" under assistant turns.
   */
  turns?: LiveRagTurn[];
  /**
   * Additional CSS class names to apply to the conversation container.
   */
  className?: string;
}

/**
 * A chat transcript component that displays a conversation between the user and agent.
 * Shows messages with timestamps and origin indicators, plus a thinking indicator
 * when the agent is processing.
 *
 * @extends ComponentProps<'div'>
 *
 * @example
 * ```tsx
 * <AgentChatTranscript
 *   agentState={agentState}
 *   messages={chatMessages}
 * />
 * ```
 */
export function AgentChatTranscript({
  agentState,
  messages = [],
  turns = [],
  className,
  ...props
}: AgentChatTranscriptProps) {
  const fallbackMessages = turns.flatMap((turn) =>
    [turn.user_message, turn.assistant_message]
      .filter((message): message is NonNullable<typeof message> => Boolean(message?.content))
      .map((message) => ({
        id: `${turn.turn_index}-${message.role}-${message.timestamp ?? message.content}`,
        timestamp: message.timestamp ?? new Date().toISOString(),
        from: { isLocal: message.role === 'user' },
        message: message.content,
      }))
  );
  const transcriptMessages = messages.length > 0 ? messages : fallbackMessages;
  let assistantIndex = 0;

  return (
    <Conversation className={className} {...props}>
      <ConversationContent>
        {transcriptMessages.map((receivedMessage) => {
          const { id, timestamp, from, message } = receivedMessage;
          const locale = navigator?.language ?? 'zh-CN';
          const messageOrigin = from?.isLocal ? 'user' : 'assistant';
          const time = new Date(timestamp);
          const title = time.toLocaleTimeString(locale, { timeStyle: 'full' });
          const turn =
            messageOrigin === 'assistant'
              ? findTurnForMessage(message, turns, assistantIndex++)
              : undefined;

          return (
            <Message key={id} title={title} from={messageOrigin}>
              <MessageContent>
                <MessageResponse>{message}</MessageResponse>
                {messageOrigin === 'assistant' && <RagEvidence turn={turn} />}
              </MessageContent>
            </Message>
          );
        })}
        <AnimatePresence>
          {agentState === 'thinking' && <AgentChatIndicator size="sm" />}
        </AnimatePresence>
      </ConversationContent>
      <ConversationScrollButton />
    </Conversation>
  );
}
