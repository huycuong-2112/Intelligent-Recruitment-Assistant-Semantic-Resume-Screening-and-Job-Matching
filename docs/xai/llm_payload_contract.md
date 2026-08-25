# Compact future LLM explanation payload (`llm_payload_v1`)

Derived deterministically from the full `xai_v1` object. Keep only role identity, final score/status, score breakdown, selected material strengths and gaps, selected experience evidence, education result, semantic score/scope, interview triggers, and referenced evidence excerpts.

Exclude vectors, tensor values, unused chunks, full raw CV/JD text, duplicate source text, and debug metadata. The future renderer may produce prose but must treat all supplied facts, scores, statuses, and evidence references as immutable.
