# OrbitDesk AI Support Agent — Architecture

This document describes the architecture of the fully offline AI support agent,
its component map, and the flow of data through the pipeline.

---

## High-Level Overview

The system is a **stateful pipeline** orchestrated by **LangGraph**. Each node
is a focused, testable unit. The user's question enters at the top, flows
through the nodes, and exits as a structured JSON response — always grounded in
the local knowledge base.

```
                          ┌─────────────────────────────────────────────┐
                          │                 USER QUESTION               │
                          └──────────────────────┬──────────────────────┘
                                                 │
                                                 ▼
                          ┌─────────────────────────────────────────────┐
                          │                    TRIAGE                    │
                          │  Classifies: Answerable / Clarification /   │
                          │              Out of Scope / Escalation       │
                          └───────┬──────────────┬──────────┬───────────┘
                                  │  Answerable │          │ Other
                                  ▼             │          ▼
                          ┌─────────────────┐   │   ┌──────────────────┐
                          │    RETRIEVAL    │   │   │    FORMATTER     │
                          │  FAISS Top-K    │   │   │  (safe response) │
                          └────────┬────────┘   │   └────────┬─────────┘
                                   │             │          (safe / escalated)
                                   ▼             │             │
                          ┌─────────────────┐   │             │
                          │    GENERATOR    │   │             │
                          │  local LLM      │   │             │
                          └────────┬────────┘   │             │
                                   │             │             │
                                   ▼             │             │
                          ┌─────────────────┐   │             │
                          │    VERIFIER     │   │             │
                          │  Passed?        │   │             │
                          └───┬─────────┬───┘   │             │
                    Passed    │         │ Failed│(retries)    │
                              │         │       /             │
                              │         ▼   ┌─────────────────┐
                              │      (retry) │  GENERATOR      │ (once)
                              │              └─────────────────┘
                              │         │
                              │         ▼ (no retries)
                              │  ┌─────────────────┐
                              │  │    FORMATTER    │ (safe failure)
                              │  └────────┬────────┘
                              │           │
                              ▼           ▼
                          ┌─────────────────────────────────────────────┐
                          │              FINAL OUTPUT (JSON)            │
                          └─────────────────────────────────────────────┘
```

---

## The Nodes

### 1. Triage Node (`nodes/triage.py`)
Decides whether the question can be answered at all.

- **Answerable** → routes to Retrieval.
- **Clarification Required** → asks a follow-up (safe response).
- **Out of Scope** → returns a safe refusal.
- **Escalation Required** → flags for a human agent.

Uses keyword rules + a semantic-similarity fallback against known OrbitDesk
topics. Fast and deterministic; no LLM call.

### 2. Retrieval Node (`nodes/retrieval.py`)
Searches **only** the local FAISS index.

- Converts the question into an embedding.
- Returns Top-K documents with content + metadata.
- Applies a similarity threshold to drop irrelevant results.
- Can never return outside content (index is built only from the KB).

### 3. Generator Node (`nodes/generator.py`)
Calls the **local** Hugging Face model (Flan-T5).

- Builds a *grounded* prompt from retrieved docs.
- Enforces "use ONLY context; never outside knowledge".
- Says explicitly when the answer isn't in the KB.

### 4. Verifier Node (`nodes/verifier.py`)
Ensures the answer is supported by the retrieved docs.

- Checks lexical support and consistency.
- Optionally runs an LLM verdict.
- On failure triggers a **single retry** of the generator.
- On repeated failure → safe fallback (no guessing).

### 5. Formatter Node (`nodes/formatter.py`)
Shapes the final output.

- Computes confidence level.
- Resolves final status.
- Produces the schema-conforming JSON.

---

## Data Flow

State is a shared dict (`state.AnswerState`) passed through LangGraph. Each
node returns a partial update merged into the state.

| Field                 | Written By      | Purpose                                    |
|-----------------------|-----------------|--------------------------------------------|
| `question`            | app             | The user's question.                       |
| `triage_label`        | triage          | Classification result.                     |
| `follow_up_question`  | triage          | Clarification prompt (if any).             |
| `documents`           | retrieval       | Top-K retrieved docs (content + metadata). |
| `sources`             | retrieval       | Filenames grounding the answer.            |
| `answer`              | generator       | Generated answer text.                     |
| `verification`        | verifier        | Passed / Failed / Not Applicable.          |
| `retry_count`         | verifier        | Number of retries performed.               |
| `confidence`          | formatter       | High / Medium / Low.                       |
| `status`              | formatter       | Success / Safe Response / Escalated / Failed. |
| `raw_output`          | formatter       | Final schema-conforming output.            |

---

## Component Map

```
AI-Support-Agent/
├── app.py                  CLI entry point
├── config.py               Central configuration (paths, models, thresholds)
├── state.py                Typed state + data structures
├── graph.py                LangGraph orchestration + routing
├── output_schema.json      Expected output shape
├── knowledge_base/         Local Markdown documentation (source of truth)
├── resolved_cases.json     Resolved support cases (extra retrieval context)
├── sample_questions.json   Seed questions for --sample
├── models/
│   └── llm.py              Local Hugging Face LLM wrapper
├── nodes/
│   ├── triage.py           Classification
│   ├── retrieval.py        FAISS search
│   ├── generator.py        Answer generation
│   ├── verifier.py         Answer verification
│   └── formatter.py        Output shaping
└── utils/
    ├── embeddings.py       Sentence embedding wrapper
    ├── vector_store.py     FAISS index + search
    ├── loader.py           Knowledge base loading
    ├── prompts.py          Prompt templates
    └── logger.py           Centralised logging
```

---

## Why This Architecture?

1. **No hallucination** — the LLM is grounded in retrieved evidence.
2. **Fully offline** — local models + local vector DB; zero cloud calls.
3. **Safe** — out-of-scope and unclear questions are handled explicitly.
4. **Reliable** — verification with retry catches and corrects bad answers.
5. **Introspectable** — every step produces auditable state and logs.
6. **Modular & testable** — each node is a pure, unit-testable function.
