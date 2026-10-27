# Solution: Autonomous TAM Support Triage Agent

## 1. Design

The triage system is designed as an autonomous, evidence-driven ReAct agent. Instead of a rigid linear pipeline, the agent actively inspects ticket metadata, decides which tools to call, inspects intermediate results, and stops only when sufficient evidence has been gathered.

```text
Incoming Ticket
      │
      ▼
Context Initialization (Cached In-Memory Indexes)
  ├── Account Indexes: O(1) by ID, lowercase domain, and company name
  ├── Ticket Partition: Grouped by account_id for fast scoped lookups
  ├── Usage Pre-computation: 3-month rolling active user trend & churn risk
  └── Knowledge Base: BM25 index over articles, metadata, and error codes
      │
      ▼
Autonomous Triage Loop (Max 15 steps)
  ├── 1. LLM analyzes context & emits tool call(s) (supports batched 1-4 calls)
  ├── 2. Tools execute: lookup account, check usage, search KB, check duplicates
  ├── 3. Deprecation check: automatically warns & directs agent to read superseded articles
  ├── 4. Tool outputs returned as compact JSON/text payloads
  └── 5. LLM stops when confident, emitting ```submit with schema-validated decision
      │
      ▼
Result Normalizer & Validator (tam_agent/schema.py)
  └── Guarantees schema adherence, valid enums, bounds, and reasoning trace
```

### Key Tools
- `lookup_account(account_id, domain, company_name)`: O(1) lookup by exact ID, domain, or company name, falling back to SequenceMatcher fuzzy matching with ambiguity detection.
- `get_usage_trend(account_id)`: Returns pre-computed 3-month vs prior 3-month active user trajectory, trend (`declining`, `stable`, `growing`), and `churn_risk` boolean (>25% drop).
- `search_tickets(account_id, lookback_days, module, error_code)`: Retrieves recent open/pending tickets scoped to the account.
- `check_duplicate(account_id, module, error_code, subject, created_at)`: Evaluates open/pending tickets within a 7-day window using module, error code, and subject similarity scoring.
- `search_kb(query, module, error_code)`: BM25 ranking across titles, bodies, and tags, returning top-4 concise candidate snippets.
- `read_kb_article(article_id)`: Fetches full article content with an explicit alert if deprecated, instructing the agent to follow the `superseded_by` successor.

### Stopping Mechanism & Guardrails
- **Evidence-Driven Exit:** The LLM terminates execution by generating a ` ```submit ` block matching `tam_agent/schema.py`.
- **Precedence Guardrail:** If an LLM response contains both a tool call and a speculative submission, tool execution takes strict precedence to prevent premature exit without complete evidence.
- **Batched Execution:** Supports calling up to 4 independent tools in parallel in a single turn, reducing latency.
- **Fail-Safe Fallback:** If `MAX_STEPS` is reached without submission, a forced structured decision is requested; if that fails, a deterministic clarification-request fallback is output to maintain 100% schema compliance.

---

## 2. Evaluation

To rigorously measure triage quality prior to production deployment:

1. **Golden Benchmark Accuracy:** Measure F1-score across `category`, `priority`, and `routed_to` against a human-verified dataset of historical tickets covering subtle multi-hop scenarios.
2. **Policy & Escalation Compliance Rate:** 100% test coverage asserting that high-tier accounts (Enterprise/Strategic), contract renewals (<90 days), and usage declines (>20%) strictly trigger priority bumps and `escalate_to_tam=true`.
3. **Multi-Hop KB Traversal Fidelity:** Verify that whenever a query hits a deprecated KB article, the agent successfully executes `read_kb_article` on the `superseded_by` successor and cites the active article ID.
4. **Duplicate Detection Precision/Recall:** Benchmark duplicate identification against historical ticket pairs tagged as duplicates within the 7-day lookback window.
5. **Account Resolution Certainty:** Ensure tickets with missing account IDs correctly resolve unambiguous domains, while genuinely ambiguous company names properly trigger `needs_clarification=true` instead of hallucinating accounts.
6. **Hallucination & Schema Validation:** Automated regression testing confirming 100% schema conformity with zero invalid enum values or out-of-bounds confidence scores.
7. **Human-in-the-Loop Shadow Evaluation:** Run the agent in shadow mode alongside TAMs for 2 weeks, evaluating customer reply helpfulness and internal note quality using a 1–5 rubric.
8. **Operational Latency & Cost Tracking:** Monitor P95 response latency (<15s) and average token consumption per ticket to ensure cost efficiency at scale.

---

## 3. Scale: 10x Data (30,000 Tickets, 4,000 KB Articles)

### Current System Behavior at 10x
- **Memory & Latency:** In-memory partition dictionaries (`tickets_by_account_id`, `account_by_id`) and BM25 index consume only ~25MB RAM. 30,000 tickets partitioned across 200 accounts average only ~150 tickets per account, keeping scoped filtering instantaneous.
- **Prompt Size & Cost:** Bounded context size. Tool outputs return concise JSON payloads (`tool_output[:1500]`), and usage trends are pre-computed summaries rather than raw rows.
- **Retrieval Quality:** Pure lexical BM25 over 4,000 articles will suffer from vocabulary mismatch and synonym misses when ticket descriptions describe symptoms differently from KB article titles.

### Architectural Changes for 10x+ Scale
1. **Hybrid Retrieval (Dense + Sparse):** Replace in-memory BM25 with a vector database (e.g., Qdrant or pgvector) combining dense semantic embeddings with BM25 via Reciprocal Rank Fusion (RRF).
2. **Persistent Partitioned Storage:** Transition from CSVs to PostgreSQL with indexes on `(account_id, created_at, status)` and Trigram/GIN indexing on company names and ticket subjects.
3. **Semantic Caching & Fast Routing:** Implement Redis caching for account health and top policy articles. Use a fast SLM classifier to immediately route routine FAQ tickets, reserving multi-turn agentic loops for complex cases.
4. **Hierarchical Multi-Agent / Small-to-Large Model Split:** Use a lightweight, high-throughput model (e.g., 8B/mini) for initial tool dispatch and account resolution, elevating to frontier reasoning models only for high-risk TAM escalations.

---

## 4. Limitations

- **Lexical BM25 Limitations:** Keyword-based KB retrieval cannot capture paraphrased customer terminology or cross-lingual queries without dense semantic embeddings.
- **Static Time Context:** Relies on a hardcoded reference date (`2026-10-20`) rather than dynamic production timestamps.
- **Stateless Feedback:** The agent does not learn online from human reviewer corrections or downstream ticket re-openings.
- **Fixed Similarity Thresholds:** Duplicate detection and fuzzy company matching rely on heuristic similarity ratios (`difflib.SequenceMatcher`) that may require fine-tuning for noisy OCR/email signatures.

---

## 5. How to Run

### Setup Environment
```bash
# Activate virtual environment
.\.venv\Scripts\Activate.ps1    # Windows PowerShell
# source .venv/bin/activate     # Linux / macOS

# Install dependencies (if setting up fresh)
pip install -r requirements.txt
```

### Validate Dataset Integrity
```bash
python scripts/validate_data.py
```

### Run Review Tickets
```bash
# Runs the 10 benchmark review tickets and writes outputs/
python run.py --set review
```

### Verify Output Compliance
```bash
python scripts/check_outputs.py
```
*(Should display `10/10 review tickets valid`)*
