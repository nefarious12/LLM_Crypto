"""Project 6: guardrails around the RAG app.

Layers, in the order a request passes through them:
  1. Input checks      length limit, prompt-injection patterns, investment-advice policy
  2. PII redaction     emails, cards (Luhn-checked), SSN, IBAN, PAN, phone numbers
  3. Untrusted content retrieved documents are scanned and quarantined if they contain instructions
  4. Output checks     answers must cite a source, and are re-scanned for PII
  5. Audit log         JSON lines with a query hash and redacted text, never raw PII
  6. Action gate       high-risk actions need approval from a different person

Pattern matching is a first line of defence, not a complete one. See redteam.py for what it misses.
Educational only. Not investment advice.
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .rag import DATA_DIR, NOT_FOUND, Retriever, answer, build_chunks

ROOT = Path(__file__).resolve().parent.parent
MAX_QUERY_CHARS = 1000

# ---------- PII ----------
PII_PATTERNS = [
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("CARD", re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    ("PHONE", re.compile(r"(?<![\d-])(?:\+?\d{1,3}[ -]?)?\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}(?![\d-])")),
]


def luhn_ok(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
    return total % 10 == 0


def redact_pii(text: str) -> tuple[str, list[str]]:
    found: list[str] = []
    for label, pattern in PII_PATTERNS:
        def sub(m, label=label):
            if label == "CARD" and not luhn_ok(m.group()):
                return m.group()
            found.append(label)
            return f"[{label}]"
        text = pattern.sub(sub, text)
    return text, found


# ---------- Prompt injection ----------
INJECTION_PATTERNS = [
    r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|all|your)\b.{0,30}\b(instructions?|rules?|prompts?|guidelines?)\b",
    r"\b(ignore|disregard|forget|override)\b.{0,30}\b(instructions?|rules?|guidelines?)\b",
    r"\b(reveal|show|print|repeat|leak)\b.{0,30}\b(system prompt|hidden prompt|your prompt|your instructions)\b",
    r"\byou are now\b",
    r"\b(developer mode|jailbreak|do anything now|\bDAN\b)",
    r"\b(act as|pretend (to be|you are)|roleplay as)\b.{0,60}\b(without|no|free of)\b.{0,15}\b(restrictions|rules|limits|filters)\b",
    r"\bsend\b.{0,80}\bto\b.{0,40}\S+@\S+",
]
_INJECTION = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in INJECTION_PATTERNS]
_ADVICE = re.compile(
    r"\b(should i|do you recommend|is it a good (idea|time) to)\b.{0,40}\b(buy|sell|invest|short)\b|\b(buy|sell) (this|the) (stock|share)s?\b",
    re.IGNORECASE,
)


def detect_injection(text: str) -> bool:
    return any(p.search(text) for p in _INJECTION)


# ---------- Audit log ----------
class AuditLog:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else ROOT / "logs" / "audit.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, **event) -> None:
        event = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **event}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


# ---------- Guarded pipeline ----------
class GuardedRAG:
    def __init__(self, doc_paths: list[Path] | None = None, audit: AuditLog | None = None):
        paths = doc_paths or [DATA_DIR / "meridian_fy2024.txt", DATA_DIR / "poisoned_doc.txt"]
        chunks = build_chunks(paths)
        self.quarantined = [c for c in chunks if detect_injection(c.text)]
        bad = {c.id for c in self.quarantined}
        self.retriever = Retriever([c for c in chunks if c.id not in bad])
        self.audit = audit or AuditLog()

    def _finish(self, decision, reason, safe_query, query, user, pii, answer_text="", sources=None):
        self.audit.write(
            user=user,
            query_hash=hashlib.sha256(query.encode()).hexdigest()[:12],
            query_redacted=safe_query,
            decision=decision,
            reason=reason,
            pii_types=sorted(set(pii)),
            sources=sources or [],
            quarantined_chunks=len(self.quarantined),
        )
        return {"decision": decision, "reason": reason, "answer": answer_text, "pii_redacted": sorted(set(pii)), "sources": sources or []}

    def ask(self, query: str, user: str = "analyst") -> dict:
        safe_query, pii = redact_pii(query[:MAX_QUERY_CHARS])
        if len(query) > MAX_QUERY_CHARS:
            return self._finish("blocked", "query too long", safe_query, query, user, pii)
        if detect_injection(query):
            return self._finish("blocked", "possible prompt injection", safe_query, query, user, pii)
        if _ADVICE.search(query):
            msg = "I can explain what the documents say, but I can't give investment advice."
            return self._finish("refused", "investment advice is out of scope", safe_query, query, user, pii, msg)
        result = answer(safe_query, self.retriever, use_llm=False)
        if result["abstained"]:
            return self._finish("abstained", "no supporting passage", safe_query, query, user, pii, NOT_FOUND)
        if not result["citations"]:
            return self._finish("abstained", "answer had no citation", safe_query, query, user, pii, NOT_FOUND)
        text, out_pii = redact_pii(result["answer"])
        return self._finish("answered", "ok", safe_query, query, user, pii + out_pii, text, result["sources"])


# ---------- Human approval for risky actions ----------
HIGH_RISK = {"send_email", "transfer_funds", "delete_record"}


class ActionGate:
    """High-risk actions wait for approval by someone other than the requester."""

    def __init__(self, audit: AuditLog | None = None):
        self.pending: dict[str, dict] = {}
        self.audit = audit or AuditLog()

    def request(self, action: str, params: dict, requested_by: str) -> dict:
        if action not in HIGH_RISK:
            self.audit.write(event="action", action=action, status="executed", by=requested_by)
            return {"status": "executed", "action": action}
        action_id = uuid.uuid4().hex[:8]
        self.pending[action_id] = {"action": action, "params": params, "requested_by": requested_by}
        self.audit.write(event="action", action=action, status="pending_approval", id=action_id, by=requested_by)
        return {"status": "pending_approval", "id": action_id}

    def approve(self, action_id: str, approver: str) -> dict:
        item = self.pending.get(action_id)
        if item is None:
            return {"status": "rejected", "reason": "unknown id"}
        if approver == item["requested_by"]:
            self.audit.write(event="action", id=action_id, status="rejected", reason="self-approval", by=approver)
            return {"status": "rejected", "reason": "requester cannot approve their own action"}
        del self.pending[action_id]
        self.audit.write(event="action", action=item["action"], id=action_id, status="executed", approved_by=approver)
        return {"status": "executed", "action": item["action"]}
