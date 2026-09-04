import assert from 'node:assert/strict';
import test from 'node:test';
import {
  type ConsultationMessage,
  createConversationIdentity,
  mergeConversationTimeline,
  resolveVoiceConnectionState,
  switchConsultationMode,
} from '../lib/medical-conversation.ts';

test('creates correlatable text and voice child identities', () => {
  const identity = createConversationIdentity('123e4567-e89b-12d3-a456-426614174000');
  assert.equal(identity.conversationId, 'conv_123e4567e89b12d3a456426614174000');
  assert.equal(identity.textSessionId, `web_${identity.conversationId}`);
  assert.equal(identity.voiceClientId, identity.conversationId);
});

test('switching input mode preserves the existing timeline', () => {
  const messages = [{ id: 'm1' }];
  const next = switchConsultationMode({ inputMode: 'text' as const, messages }, 'voice');
  assert.equal(next.inputMode, 'voice');
  assert.equal(next.messages, messages);
});

test('merges text and voice turns and keeps persisted evidence', () => {
  const base: ConsultationMessage = {
    id: 'live-1',
    role: 'assistant',
    channel: 'voice',
    content: '建议尽快就医。',
    createdAt: '2026-09-04T08:00:02.000Z',
  };
  const timeline = mergeConversationTimeline(
    [
      {
        id: 'text-1',
        role: 'user',
        channel: 'text',
        content: '我有些头晕',
        createdAt: '2026-09-04T08:00:00.000Z',
      },
    ],
    [base, { ...base, id: 'turn-1', evidence: [{ id: 'e1', title: '门诊记录' }] }]
  );
  assert.deepEqual(
    timeline.map((message) => message.id),
    ['text-1', 'turn-1']
  );
  assert.equal(timeline[1]?.evidence?.[0]?.title, '门诊记录');
});

test('keeps identical content when it came from different input channels', () => {
  const shared = {
    role: 'user' as const,
    content: '我有些头晕',
    createdAt: '2026-09-04T08:00:00.000Z',
  };
  const timeline = mergeConversationTimeline(
    [{ ...shared, id: 'text-1', channel: 'text' }],
    [{ ...shared, id: 'voice-1', channel: 'voice' }]
  );
  assert.equal(timeline.length, 2);
});

test('maps key voice states to explicit user states', () => {
  assert.equal(resolveVoiceConnectionState(undefined, false, false), 'disconnected');
  assert.equal(resolveVoiceConnectionState(undefined, false, true), 'connecting');
  assert.equal(resolveVoiceConnectionState('thinking', true, false), 'thinking');
  assert.equal(resolveVoiceConnectionState('speaking', true, false), 'speaking');
  assert.equal(resolveVoiceConnectionState('listening', true, false), 'listening');
});
