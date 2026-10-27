"""Tools the triage agent can call. Each tool takes simple string/dict args
and returns a concise text summary the LLM can reason over.

The Context object (built once per run) holds all indexed data.
"""
from __future__ import annotations
import re
from datetime import datetime, timedelta
from math import log
from collections import Counter
from difflib import SequenceMatcher

from .data import AS_OF_DATE

# ── helpers ──────────────────────────────────────────────────────────

TODAY = datetime.fromisoformat(AS_OF_DATE)


def _days_until(date_str: str) -> int | None:
    if not date_str:
        return None
    try:
        return (datetime.fromisoformat(date_str) - TODAY).days
    except ValueError:
        return None


def _normalize(text: str) -> str:
    """Lower-case, strip common suffixes for fuzzy matching."""
    t = text.lower().strip()
    for suf in (" inc", " inc.", " llc", " ltd", " corp", " corp.",
                " group", " co", " co.", " gmbh", " plc", " sa"):
        if t.endswith(suf):
            t = t[: -len(suf)].strip()
    return t


# ── BM25 index (built once) ─────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9](?:[a-z0-9\-]*[a-z0-9])?", text.lower())


class BM25Index:
    """Lightweight BM25 (Okapi) over KB article metadata + body."""

    def __init__(self, articles: list[dict]):
        self.articles = articles
        self.corpus: list[list[str]] = []
        for art in articles:
            tokens = _tokenize(
                f"{art['meta'].get('title', '')} "
                f"{art['meta'].get('module', '')} "
                f"{' '.join(art['meta'].get('error_codes', []) if isinstance(art['meta'].get('error_codes'), list) else [])} "
                f"{art['meta'].get('type', '')} "
                f"{art['body']}"
            )
            self.corpus.append(tokens)
        self.n = len(self.corpus)
        self.avgdl = sum(len(d) for d in self.corpus) / max(self.n, 1)
        # IDF
        self.idf: dict[str, float] = {}
        df: Counter = Counter()
        for doc in self.corpus:
            for t in set(doc):
                df[t] += 1
        for term, freq in df.items():
            self.idf[term] = log((self.n - freq + 0.5) / (freq + 0.5) + 1)

    def query(self, q: str, top_k: int = 8) -> list[tuple[dict, float]]:
        tokens = _tokenize(q)
        if not tokens:
            return []
        scores = [0.0] * self.n
        k1, b = 1.5, 0.75
        for i, doc in enumerate(self.corpus):
            dl = len(doc)
            tf_map: Counter = Counter(doc)
            for t in tokens:
                if t not in self.idf:
                    continue
                tf = tf_map.get(t, 0)
                numer = tf * (k1 + 1)
                denom = tf + k1 * (1 - b + b * dl / self.avgdl)
                scores[i] += self.idf[t] * numer / denom
        ranked = sorted(range(self.n), key=lambda i: scores[i], reverse=True)
        results = []
        for i in ranked[:top_k]:
            if scores[i] > 0:
                results.append((self.articles[i], round(scores[i], 3)))
        return results


# ── Tool functions ───────────────────────────────────────────────────

def lookup_account(ctx, *, account_id: str = "", domain: str = "",
                   company_name: str = "") -> str:
    """Resolve an account by account_id, domain, or fuzzy company name.
    Returns account details or 'not found'."""
    accts = ctx.accounts

    # 1. Direct ID lookup
    if account_id:
        row = accts[accts["account_id"] == account_id]
        if not row.empty:
            return _format_account(row.iloc[0])
        return f"No account found with id={account_id}"

    # 2. Domain lookup (exact)
    if domain:
        row = accts[accts["domain"] == domain]
        if not row.empty:
            return _format_account(row.iloc[0])
        return f"No account found with domain={domain}"

    # 3. Fuzzy company name
    if company_name:
        norm = _normalize(company_name)
        candidates = []
        for _, r in accts.iterrows():
            score = SequenceMatcher(None, norm, _normalize(r["company_name"])).ratio()
            if score > 0.55:
                candidates.append((score, r))
        candidates.sort(key=lambda x: -x[0])
        if not candidates:
            return f"No account found matching company_name='{company_name}'"
        if len(candidates) == 1 or candidates[0][0] > 0.85:
            return _format_account(candidates[0][1])
        # Multiple ambiguous matches
        lines = [f"Ambiguous matches for '{company_name}':"]
        for score, r in candidates[:5]:
            lines.append(f"  - {r['account_id']} {r['company_name']} (domain={r['domain']}, score={score:.2f})")
        return "\n".join(lines)

    return "Error: provide at least one of account_id, domain, or company_name."


def _format_account(r) -> str:
    renewal_days = _days_until(r.get("renewal_date", ""))
    renewal_info = f"{r['renewal_date']} ({renewal_days} days away)" if renewal_days is not None else r.get("renewal_date", "N/A")
    return (
        f"Account: {r['account_id']} | {r['company_name']}\n"
        f"  Domain: {r['domain']}\n"
        f"  Tier: {r['tier']} | ARR: {r['arr_band']} | Industry: {r['industry']} | Region: {r['region']}\n"
        f"  Renewal: {renewal_info}\n"
        f"  Modules: {r['modules_licensed']}\n"
        f"  SSO enabled: {r.get('sso_enabled_date', 'N/A')}\n"
        f"  Open escalations: {r['open_escalations']}\n"
        f"  Last QBR: {r.get('last_qbr_date', 'N/A')}\n"
        f"  NPS: {r.get('nps_last', 'N/A')}"
    )


def get_usage_trend(ctx, *, account_id: str) -> str:
    """Get 6-month usage trend for an account. Highlights drops >20%."""
    usage = ctx.usage
    rows = usage[usage["account_id"] == account_id].sort_values("month")
    if rows.empty:
        return f"No usage data for {account_id}"

    lines = ["Month      | Active Users | Docs Processed | Login Rate % | Tickets"]
    lines.append("-" * 70)
    for _, r in rows.tail(6).iterrows():
        lines.append(
            f"{r['month']}   | {r['active_users']:>12} | {r['documents_processed']:>14} | "
            f"{r['login_rate_pct']:>12} | {r['tickets_opened']}"
        )

    # Compute trend over last 3 months vs prior 3 months
    recent = rows.tail(6)
    if len(recent) >= 6:
        last3 = recent.tail(3)["active_users"].mean()
        prev3 = recent.head(3)["active_users"].mean()
        if prev3 > 0:
            pct = (last3 - prev3) / prev3 * 100
            trend = f"DECLINING ({pct:+.1f}%)" if pct < -20 else (
                f"GROWING ({pct:+.1f}%)" if pct > 10 else f"STABLE ({pct:+.1f}%)")
            lines.append(f"\nUser trend (3mo vs prior 3mo): {trend}")
            if pct < -25:
                lines.append("[!] ALERT: Active users declined >25% — account risk override may apply per KB-2914")
    return "\n".join(lines)


def search_tickets(ctx, *, account_id: str, lookback_days: int = 14,
                   module: str = "", error_code: str = "") -> str:
    """Search recent tickets for an account. Optionally filter by module/error_code."""
    tickets = ctx.tickets
    mask = tickets["account_id"] == account_id
    if not mask.any():
        # Try by domain
        return f"No tickets found for account {account_id}"

    subset = tickets[mask].copy()
    subset["_created"] = subset["created_at"].apply(
        lambda x: datetime.fromisoformat(x) if x else datetime.min)
    cutoff = TODAY - timedelta(days=lookback_days)
    subset = subset[subset["_created"] >= cutoff]

    if module:
        mod_lower = module.lower()
        subset = subset[subset["module"].str.lower().str.contains(mod_lower, na=False)]
    if error_code:
        ec = error_code.upper()
        subset = subset[
            (subset["error_code"].str.upper() == ec) |
            subset["description"].str.contains(ec, case=False, na=False)
        ]

    if subset.empty:
        return f"No matching tickets for {account_id} in last {lookback_days} days"

    subset = subset.sort_values("_created", ascending=False).head(10)
    lines = [f"Found {len(subset)} recent tickets for {account_id}:"]
    for _, t in subset.iterrows():
        lines.append(
            f"  {t['ticket_id']} | {t['created_at']} | {t['status']} | "
            f"module={t['module']} | error={t['error_code']} | {t['subject']}"
        )
    return "\n".join(lines)


def search_kb(ctx, *, query: str, module: str = "",
              error_code: str = "", article_type: str = "") -> str:
    """Search KB articles by text query. Optionally filter by module, error_code, or type."""
    # Build a combined query
    q_parts = [query]
    if module:
        q_parts.append(module)
    if error_code:
        q_parts.append(error_code)
    if article_type:
        q_parts.append(article_type)
    combined = " ".join(q_parts)

    results = ctx.kb_index.query(combined, top_k=8)

    # Post-filter by exact error_code if specified
    if error_code:
        ec = error_code.upper()
        # Prioritize exact error code matches
        exact = [(art, sc) for art, sc in results
                 if ec in (art["meta"].get("error_codes", []) if isinstance(art["meta"].get("error_codes"), list) else [])
                 or ec in art.get("body", "")]
        others = [(art, sc) for art, sc in results if (art, sc) not in exact]
        results = exact + others

    if module:
        mod_lower = module.lower()
        # Boost exact module matches
        exact = [(art, sc) for art, sc in results if art["meta"].get("module", "").lower() == mod_lower]
        others = [(art, sc) for art, sc in results if (art, sc) not in exact]
        results = exact + others

    if not results:
        return "No KB articles found."

    lines = [f"Top {len(results)} KB results:"]
    for art, score in results[:8]:
        meta = art["meta"]
        status_flag = " [DEPRECATED]" if meta.get("status") == "deprecated" else ""
        superseded = f" -> superseded by {meta['superseded_by']}" if meta.get("superseded_by") else ""
        lines.append(
            f"  {meta['id']} (score={score}) | {meta.get('title', '')} | "
            f"module={meta.get('module', '')} | type={meta.get('type', '')} | "
            f"status={meta.get('status', '')}{status_flag}{superseded}"
        )
    return "\n".join(lines)


def read_kb_article(ctx, *, article_id: str) -> str:
    """Read the full content of a specific KB article."""
    art = ctx.kb_by_id.get(article_id)
    if not art:
        return f"Article {article_id} not found."

    meta = art["meta"]
    lines = [f"=== {meta['id']}: {meta.get('title', '')} ==="]
    lines.append(f"Module: {meta.get('module', '')} | Type: {meta.get('type', '')} | Status: {meta.get('status', '')}")
    if meta.get("error_codes"):
        ec = meta["error_codes"] if isinstance(meta["error_codes"], list) else [meta["error_codes"]]
        lines.append(f"Error codes: {', '.join(ec)}")
    if meta.get("superseded_by"):
        lines.append(f"[DEPRECATED] — superseded by: {meta['superseded_by']}")
    if meta.get("related"):
        rel = meta["related"] if isinstance(meta["related"], list) else [meta["related"]]
        lines.append(f"Related: {', '.join(rel)}")
    lines.append(f"Last updated: {meta.get('last_updated', 'unknown')}")
    lines.append("")
    lines.append(art["body"].strip())
    return "\n".join(lines)


def check_duplicate(ctx, *, account_id: str, module: str,
                    error_code: str, subject: str,
                    created_at: str) -> str:
    """Check if a ticket is a duplicate of an existing open/pending ticket
    for the same account, module, and error code within 7 days."""
    tickets = ctx.tickets
    mask = (tickets["account_id"] == account_id) & (
        tickets["status"].isin(["open", "pending_customer"]))

    subset = tickets[mask].copy()
    if subset.empty:
        return "No open/pending tickets for this account."

    try:
        ticket_date = datetime.fromisoformat(created_at)
    except (ValueError, TypeError):
        ticket_date = TODAY
    window = ticket_date - timedelta(days=7)

    subset["_created"] = subset["created_at"].apply(
        lambda x: datetime.fromisoformat(x) if x else datetime.min)
    subset = subset[subset["_created"] >= window]
    subset = subset[subset["_created"] < ticket_date]

    if subset.empty:
        return "No potential duplicates found within the 7-day window."

    # Match by module and error code
    candidates = []
    mod_lower = module.lower() if module else ""
    ec_upper = error_code.upper() if error_code else ""

    for _, t in subset.iterrows():
        score = 0
        if mod_lower and t["module"].lower().find(mod_lower) >= 0:
            score += 2
        if ec_upper and (t["error_code"].upper() == ec_upper or
                         ec_upper in t.get("description", "").upper()):
            score += 3
        # Check subject similarity
        if subject and t["subject"]:
            sim = SequenceMatcher(None, subject.lower(), t["subject"].lower()).ratio()
            if sim > 0.6:
                score += 2
        if score >= 2:
            candidates.append((score, t))

    candidates.sort(key=lambda x: -x[0])
    if not candidates:
        return "No matching duplicates found (checked module, error code, and subject similarity)."

    lines = ["Potential duplicates:"]
    for score, t in candidates[:5]:
        lines.append(
            f"  {t['ticket_id']} | {t['created_at']} | {t['status']} | "
            f"module={t['module']} | error={t['error_code']} | "
            f"subject={t['subject']} (match_score={score})"
        )
    return "\n".join(lines)


# ── Tool registry ────────────────────────────────────────────────────

TOOLS = {
    "lookup_account": {
        "function": lookup_account,
        "description": (
            "Resolve an account by account_id, domain, or company_name. "
            "Returns account details including tier, renewal date, modules, escalations."
        ),
        "parameters": {
            "account_id": "Account ID (e.g., ACC-1234). Optional.",
            "domain": "Email domain (e.g., acme.example). Optional.",
            "company_name": "Company name for fuzzy matching. Optional.",
        },
    },
    "get_usage_trend": {
        "function": get_usage_trend,
        "description": (
            "Get 6-month usage trend for an account: active users, documents processed, "
            "login rate, tickets opened. Highlights declines >20%."
        ),
        "parameters": {
            "account_id": "Account ID (required).",
        },
    },
    "search_tickets": {
        "function": search_tickets,
        "description": (
            "Search recent tickets for an account. Filter by module and/or error_code. "
            "Default lookback is 14 days."
        ),
        "parameters": {
            "account_id": "Account ID (required).",
            "lookback_days": "Number of days to look back (default 14).",
            "module": "Module name filter (optional).",
            "error_code": "Error code filter (optional).",
        },
    },
    "search_kb": {
        "function": search_kb,
        "description": (
            "Search the knowledge base using text query. Optionally filter by module, "
            "error_code, or article_type (troubleshooting, how_to, limit, known_issue, "
            "policy, reference, faq, release_notes). Returns top matches with relevance scores. "
            "If a result is marked DEPRECATED, you MUST read its superseding article."
        ),
        "parameters": {
            "query": "Search query text (required).",
            "module": "Module filter (optional).",
            "error_code": "Error code filter (optional).",
            "article_type": "Article type filter (optional).",
        },
    },
    "read_kb_article": {
        "function": read_kb_article,
        "description": (
            "Read the full content of a specific KB article by its ID. "
            "Use this to get detailed instructions, follow superseded_by chains, "
            "or read related articles."
        ),
        "parameters": {
            "article_id": "KB article ID (e.g., KB-1234). Required.",
        },
    },
    "check_duplicate": {
        "function": check_duplicate,
        "description": (
            "Check if a ticket might be a duplicate of an existing open/pending ticket "
            "for the same account. Matches on module, error_code, and subject similarity "
            "within a 7-day window."
        ),
        "parameters": {
            "account_id": "Account ID (required).",
            "module": "Module of the current ticket (required).",
            "error_code": "Error code of the current ticket (may be empty).",
            "subject": "Subject line of the current ticket.",
            "created_at": "Creation timestamp of the current ticket.",
        },
    },
}
