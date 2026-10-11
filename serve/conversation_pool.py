"""Stable conversation identity for the opt-in native context pool.

Only identity metadata is inspected. Request IDs, user/account IDs and prompt_cache_key
do not identify a conversation. Raw IDs and prompt contents are never logged here.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import OrderedDict


SHORT_INPUT_LIMIT = 3000
SHORT_INPUT_RATIO = 4


class InputRouter:
    """Divert unusually short side requests without replacing a named input baseline.

    The service serializes pool requests. Keep only as many recent named baselines as
    the pool has named slots, and reset them with the engine. Counts are API input
    tokens, excluding output and the tokens added by internal continuations.
    """

    def __init__(self, named_slots: int):
        self.named_slots = max(0, named_slots)
        self.inputs: OrderedDict[str, int] = OrderedDict()

    def route(self, key: str, input_tokens: int) -> tuple[str, int]:
        """Return (effective key, previous named input length); empty key means anonymous."""
        previous = self.inputs.get(key, 0)
        if (key and 0 < input_tokens < SHORT_INPUT_LIMIT and previous >= SHORT_INPUT_LIMIT
                and input_tokens * SHORT_INPUT_RATIO <= previous):
            return "", previous
        if key and self.named_slots:
            self.inputs[key] = input_tokens
            self.inputs.move_to_end(key)
            while len(self.inputs) > self.named_slots:
                self.inputs.popitem(last=False)
        return key, previous


def _id(value):
    return value if isinstance(value, str) and value.strip() and len(value) <= 1024 else None


def _object(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except ValueError:
            pass
    return {}


def identity(request: dict) -> tuple[str, str]:
    """Return (SHA-256 key, field source), or ('', 'anonymous').

    A subtask can share its parent's session_id, so include thread_id whenever present.
    The same session/thread pair has the same key across supported metadata encodings.
    """
    def keyed(kind, values, source):
        payload = json.dumps([kind, *values], ensure_ascii=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("ascii")).hexdigest(), source

    def session(meta, source):
        sid, tid = _id(meta.get("session_id")), _id(meta.get("thread_id"))
        if sid and tid:
            return keyed("session_thread", [sid, tid], source)
        if sid:
            return keyed("session", [sid], source)
        if tid:
            return keyed("thread", [tid], source)
        return None

    explicit = _id(request.get("strata_conversation_id"))
    if explicit:
        return keyed("conversation", [explicit], "strata_conversation_id")
    client = _object(request.get("client_metadata"))
    headers = _object(request.get("_strata_pool_headers"))
    for meta, source in ((client, "client_metadata"), (headers, "header")):
        turn = _object(meta.get("x-codex-turn-metadata"))
        # Partial Codex turn metadata must not merge all sub-agents under one parent.
        if _id(turn.get("session_id")) and _id(turn.get("thread_id")):
            return session(turn, source + ".x-codex-turn-metadata")
    metadata = _object(request.get("metadata"))
    for meta, source in ((request, "body"), (client, "client_metadata"), (metadata, "metadata")):
        cid = _id(meta.get("conversation_id"))
        if cid:
            return keyed("conversation", [cid], source + ".conversation_id")
        result = session(meta, source + ".session/thread_id")
        if result:
            return result
    cid = _id(headers.get("x-conversation-id"))
    if cid:
        return keyed("conversation", [cid], "header.x-conversation-id")
    result = session({"session_id": headers.get("x-session-id"), "thread_id": headers.get("x-thread-id")},
                     "header.session/thread_id")
    if result:
        return result
    # Claude Code versions encode the session inside metadata.user_id as JSON or a suffix.
    user = metadata.get("user_id")
    result = session(_object(user), "metadata.user_id.session_id")
    if result:
        return result
    match = re.search(r"(?:^|_)session_([a-fA-F0-9-]{16,64})$", user) if isinstance(user, str) else None
    if match:
        return keyed("session", [match.group(1)], "metadata.user_id.session_suffix")
    return "", "anonymous"


def state_line(line: str) -> dict:
    """The engine's POOL line, without exposing raw client IDs."""
    values = dict(word.split("=", 1) for word in line.split()[1:] if "=" in word)
    channels = []
    for item in values.get("channels", "-").split(","):
        if ":" not in item:
            continue
        key, tokens = item.rsplit(":", 1)
        channels.append({"id": key, "tokens": int(tokens)})
    return {"used_tokens": int(values.get("tokens", 0)), "budget_tokens": int(values.get("budget", 0)),
            "named_used": int(values.get("named", 0)), "anonymous_used": bool(int(values.get("anonymous", 0))),
            "evictions": int(values.get("evictions", 0)), "active": values.get("active"), "channels": channels}
