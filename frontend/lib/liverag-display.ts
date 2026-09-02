export function decodeLiveRagDisplayText(value?: string | null) {
  if (!value) return '';

  let decoded = value.replace(/(?:_[0-9A-Fa-f]{2}){2,}/g, (match) => match.replace(/_/g, '%'));
  for (let index = 0; index < 2; index += 1) {
    try {
      const next = decodeURIComponent(decoded);
      if (next === decoded) break;
      decoded = next;
    } catch {
      break;
    }
  }

  return decoded;
}

export function getLiveRagDisplayName(value?: string | null) {
  if (!value) return '';
  const segment = value.split(/[\\/]/).filter(Boolean).at(-1) ?? value;
  return decodeLiveRagDisplayText(segment);
}
