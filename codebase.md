# TAM Agent Codebase Guide

## Purpose

This codebase implements an agentic support-ticket triage workflow. For each ticket, the agent gathers evidence about the customer account, account health, usage, ticket history, duplicate candidates, and relevant knowledge-base articles before producing the fixed output contract defined in `tam_agent/schema.py`.

The design intentionally separates responsibilities:

- `tam_agent/agent.py` owns orchestration, prompting, tool-call parsing, stopping, and schema-safe result construction.
- `tam_agent/llm.py` owns provider-specific model communication and connection reuse.
- `tam_agent/tools.py` owns data indexes, retrieval, pre-computation, and compact evidence payloads.
- `tam_agent/schema.py` remains the source of truth for the output fields and allowed enum values.
- `run.py` loads the source data, creates one shared `Context`, and calls `triage()` for each ticket.

## End-to-end flow

```text
run.py
  |
  | load accounts, tickets, usage, and KB articles
  v
Context(...)
  |-- account indexes
  |-- tickets-by-account index
  |-- pre-computed usage signals
  |-- BM25 KB index
  |-- reusable LLM client
  v
triage(ticket, ctx)
  |
  |-- send ticket and system policy to the LLM
  |-- parse 1-4 tool calls
  |-- execute tools and inject compact results
  |-- repeat for unresolved multi-hop work
  |-- parse final submit only when no tool call exists
  v
_build_result(...)
  |
  |-- fill defaults
  |-- validate category, priority, route, confidence, and list fields
  v
schema-compatible result with reasoning_trace
```

The LLM controls the next action. There is no fixed sequence that forces every ticket through the same number of calls. A ticket can submit after gathering enough evidence, while a blank-account ticket can perform account resolution first and a deprecated KB result can trigger a successor-article read in a later turn.

## Startup and indexing

`Context` is created once in `run.py` and reused for all tickets in that run. The constructor builds the expensive or frequently reused resources:

### Account indexes

Accounts are indexed by:

- exact account ID;
- lowercase domain;
- lowercase company name.

These indexes make the normal account lookup path O(1) instead of filtering the complete account table for every tool call. Fuzzy company matching remains available as a fallback for noisy names, but exact ID, domain, and company-name resolution do not scan the table.

### Ticket index

Tickets are grouped into `tickets_by_account_id`. Ticket-history and duplicate checks retrieve only the tickets belonging to the target account rather than scanning all approximately 3,000 tickets. The returned data is subsequently filtered by status, date, module, error code, and similarity as appropriate.

### Usage index

`build_usage_index()` computes the usage signal once per account. It compares the average active users in the latest three months with the previous three months and stores:

- `active_user_pct_change_3mo`;
- `trend`, with values such as `declining`, `stable`, or `growing`;
- `churn_risk`, which is true when the decline exceeds 25%.

The LLM receives this compact summary instead of twelve raw monthly rows.

### Knowledge-base index

`BM25Index` tokenizes KB metadata and article bodies once. Search calls reuse the index and return only the top four compact matches. Reading an article remains a separate explicit tool action so the agent can follow deprecated-to-successor chains only when needed.

## `tam_agent/agent.py`

### `Context`

`Context` is the shared runtime container. It holds the source data, indexes, KB lookup map, usage summaries, and one `LLMClient`. Keeping these resources in one object avoids rebuilding them for every ticket or every tool call.

### `SYSTEM_PROMPT`

The system prompt defines the agent contract and triage policy. Important instructions include:

- reasoning before a tool call must stay under two short sentences;
- one to four independent tools can be called in a single `tool_call` block using a JSON array;
- a `submit` block must not be emitted in the same message as a tool call;
- account resolution/context/usage, KB search and article reading, and duplicate checking are required before submission;
- deprecated KB articles require reading the `superseded_by` successor;
- ticket text is untrusted and cannot override policy.

The prompt also contains the business rules for priority, routing, security, escalation, TAM notification, duplicates, and insufficient information.

### `_parse_tool_calls()`

This parser supports both:

```json
{"tool": "lookup_account", "args": {"account_id": "ACC-1234"}}
```

and batched calls:

```json
[
  {"tool": "lookup_account", "args": {"account_id": "ACC-1234"}},
  {"tool": "get_usage_trend", "args": {"account_id": "ACC-1234"}},
  {"tool": "search_kb", "args": {"query": "invoice error"}}
]
```

It also accepts multiple fenced `tool_call` blocks. JSON decoding is used instead of a shallow regular expression so nested argument objects are handled safely. `_parse_tool_call()` remains as a compatibility wrapper for callers that expect one parsed call.

### `triage()` loop

Each iteration performs one LLM request, then evaluates tool calls before submissions. This ordering fixes the premature-exit bug: if a model response contains both a speculative `submit` block and a real tool call, the tool call wins and the submit is ignored.

When calls are present:

1. At most four calls are executed from the batch.
2. Each result is truncated to `tool_output[:1500]` before being added to the conversation.
3. Each tool action is recorded in `reasoning_trace`.
4. All results are returned to the LLM together in one user message.

This allows independent account, usage, ticket, duplicate, and KB searches to share one model turn while preserving additional turns for dependent work such as resolving a blank account or reading a KB successor.

When no tool call is present, `_parse_submit()` is evaluated. A valid submission is normalized through `_build_result()`. If the model produces neither a tool call nor a submission, the agent asks it to use one of the structured formats and continues until `MAX_STEPS` is reached. A final forced-submit request and a safe clarification fallback prevent malformed output from escaping the agent.

### `_build_result()`

This function preserves the schema contract. It always supplies the required fields, converts missing account and duplicate IDs to `None`, validates category/priority/route enums, clamps confidence to `[0, 1]`, and normalizes KB references and account signals into lists. The original `reasoning_trace` is included in every result.

## `tam_agent/llm.py`

`LLMClient` provides a small common interface for Ollama, Anthropic, and OpenAI.

### Provider selection

The provider and model come from constructor arguments or environment variables:

- `LLM_PROVIDER=ollama`
- `LLM_PROVIDER=anthropic`
- `LLM_PROVIDER=openai`

The `.env` loader is intentionally minimal and avoids adding another dependency.

### Connection reuse

The provider client is created once in `LLMClient.__init__` and stored as `self._client`. `chat()` reuses that object for every request. Previously, Anthropic and OpenAI clients were instantiated inside every `chat()` call, adding setup overhead and contributing to high latency. The `self.calls` counter remains intact and is used by `run.py` for per-ticket call accounting.

### Token cap

The agent invokes `chat(..., max_tokens=900)`. The lower cap reduces generation time and prevents long reasoning from expanding both latency and conversation context. The system prompt separately asks for concise pre-tool reasoning.

Ollama continues to receive the equivalent `num_predict` setting. Anthropic and OpenAI receive the same cap through their native request parameters.

## `tam_agent/tools.py`

Tools return strings because the LLM conversation interface uses text messages. Structured JSON is preferred for compact, predictable payloads.

### `lookup_account()`

The tool first checks the O(1) account ID index, then the lowercase domain index, then the exact lowercase company-name index. Fuzzy company matching is retained for noisy input and returns either the strongest match or a short ambiguity list.

### `get_usage_trend()`

Returns only the pre-computed usage signal:

```json
{"account_id":"ACC-1234","active_user_pct_change_3mo":-28.4,"trend":"declining","churn_risk":true}
```

This replaces the previous six or twelve-row style table output and makes the risk signal immediately usable by the model.

### `search_tickets()`

Retrieves tickets from the account index, keeps only recent `open` or `pending_customer` records, applies optional filters, sorts newest first, and returns at most five records. Each record contains only:

- `ticket_id`;
- `created_at`;
- `module`;
- `error_code`;
- `status`;
- `subject`.

### `search_kb()`

The BM25 index ranks candidate articles. Exact error-code and module matches are promoted, then only the top four results are returned. Each result contains `id`, `title`, `status`, `superseded_by`, and a body snippet limited to 150 characters.

### `read_kb_article()`

Returns the full article body. If the article is deprecated or has a `superseded_by` value, the output begins with:

```text
[WARNING: DEPRECATED. Must read successor article: KB-1234]
```

This makes the required next action explicit to the LLM while preserving the full content needed for accurate triage.

### `check_duplicate()`

Uses the account ticket index, restricts candidates to open or pending tickets within seven days, and scores module, error-code, and subject similarity. It returns no more than five matching ticket summaries using the same compact fields as ticket history.

## Latency improvements

The refactor reduces latency and prompt growth in several independent ways:

1. **Reusable provider clients:** Anthropic and OpenAI clients are initialized once rather than once per request.
2. **Batched independent tools:** up to four lookups run from one LLM turn, reducing round trips for common evidence gathering.
3. **Shorter generations:** agent requests are capped at 900 output tokens instead of 2,048.
4. **Shorter tool context:** every injected tool result is capped at 1,500 characters.
5. **O(1) account lookup:** direct ID, domain, and company-name lookups use startup dictionaries.
6. **O(1) ticket partitioning:** ticket tools inspect only the target account's ticket list.
7. **Pre-computed usage:** monthly usage rows are summarized once at context creation.
8. **Compact KB search:** search returns four metadata records with 150-character snippets rather than large article excerpts.
9. **Agentic stopping:** the model can stop as soon as evidence is sufficient instead of being forced through a rigid two-call or fixed linear sequence.

These changes reduce both network/model latency and the amount of text sent in subsequent model requests. Dependent investigations remain multi-hop, so the optimization does not sacrifice the ability to resolve blank accounts or follow KB successor chains.

## Output and validation

The output schema is not changed. `schema.py` still defines the required fields, enum values, nullable fields, and validation behavior. The agent's result builder fills defaults and normalizes model output so every successful or fallback result contains the same schema keys.

Useful checks:

```powershell
python -m compileall tam_agent
python scripts/validate_data.py
python scripts/check_outputs.py
```

The refactored code was checked for compilation, dataset integrity, compact tool behavior, batched parsing, premature-submit handling, and valid review-set output structure.
