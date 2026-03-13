# ◈ Enterprise Data Insight Agent

> **Hardened Build v2.2** — Red Team Cleared · Production-Ready Streamlit AI Analytics Platform

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4.1-412991?logo=openai&logoColor=white)](https://platform.openai.com/)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063?logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![Security](https://img.shields.io/badge/Security-Red%20Team%20Cleared-success)](./README.md#security)

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Security Model](#security-model)
   - [Red Team Report — v2.2](#red-team-report--v22)
   - [Vulnerability History](#vulnerability-history)
   - [Residual Risk Register](#residual-risk-register)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Usage](#usage)
7. [Data Flow](#data-flow)
8. [API Reference](#api-reference)
9. [Performance Characteristics](#performance-characteristics)
10. [Development Guide](#development-guide)
11. [Deployment](#deployment)
12. [Changelog](#changelog)

---

## Overview

The **Enterprise Data Insight Agent** is a single-file Streamlit application that accepts CSV or Excel datasets, profiles them statistically, and returns an AI-generated executive intelligence report via the OpenAI GPT-4.1 Responses API.

**Key capabilities:**

- Upload CSV / Excel (up to 50 MB, 500,000 rows)
- Automatic column type detection and datetime inference
- Statistical profiling: numeric distributions, categorical breakdowns, missing-value analysis
- AI-generated executive summary, key metrics, insights, data quality score, recommended actions, and risk flags
- All LLM output validated through a typed Pydantic schema before rendering
- Full HTML-escape pipeline — no raw LLM content ever reaches the browser DOM
- Downloadable `.txt` report with AI-content header

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser / Streamlit UI                   │
│                                                                 │
│  ┌──────────┐   ┌──────────────────┐   ┌────────────────────┐  │
│  │ Sidebar  │   │  Dataset Preview │   │ Intelligence Report │  │
│  │ Uploader │   │  (100-row safe   │   │ (Pydantic-validated │  │
│  │ MIME +   │   │   HTML-escaped   │   │  HTML-escaped       │  │
│  │ size     │   │   view)          │   │  output)            │  │
│  │ check    │   └──────────────────┘   └────────────────────┘  │
│  └────┬─────┘                                                   │
└───────┼─────────────────────────────────────────────────────────┘
        │ UploadedFile (bytes)
        ▼
┌───────────────────────────────────┐
│         FILE LOADING LAYER        │
│  pd.read_csv / pd.read_excel      │
│  encoding="utf-8" errors=replace  │
│  engine="openpyxl" for xlsx       │
│  Row truncation at MAX_ROWS       │
│  Datetime inference (vectorised)  │
│  SHA-256 dataset fingerprint      │
└──────────────┬────────────────────┘
               │ pd.DataFrame
               ▼
┌───────────────────────────────────┐
│        PROFILING LAYER            │
│  profile_dataframe()              │
│  @st.cache_data (hash-keyed)      │
│  Column name sanitization         │
│  NaN-safe float serialization     │
│  No sample rows (PII guard)       │
└──────────────┬────────────────────┘
               │ profile dict (JSON-safe)
               ▼
┌───────────────────────────────────┐
│          AGENT LAYER              │
│  run_agent()                      │
│  Session rate limit check         │
│  Hardened system prompt           │
│  GPT-4.1 Responses API call       │
│  Response None/empty guard        │
│  Payload size gate (64 KB)        │
│  Key count gate (≤ 20 keys)       │
│  JSONDecodeError guard            │
│  AgentReport Pydantic parse       │
└──────────────┬────────────────────┘
               │ AgentReport (validated + escaped)
               ▼
┌───────────────────────────────────┐
│        RENDERING LAYER            │
│  All values html.escape()d        │
│  No raw LLM text in DOM           │
│  Structured Pydantic fields only  │
└───────────────────────────────────┘
```

---

## Security Model

### Red Team Report — v2.2

The following findings were identified during the internal red team review of v2.1 and fully mitigated in v2.2. All findings are documented with CVE-style severity ratings.

---

#### 🔴 RT-01 — Unguarded `json.loads` on LLM Output
| Field       | Detail |
|-------------|--------|
| **Severity**   | HIGH |
| **Component**  | `run_agent()` → L348 |
| **Vector**     | If the LLM returns malformed JSON (truncated, garbled, or adversarially crafted), `json.loads` raises an unhandled `json.JSONDecodeError`, propagating raw exception detail to the Streamlit error surface and potentially leaking internal path information. |
| **Impact**     | Information disclosure, unhandled crash, denial of service for the current session |
| **Fix (v2.2)** | Wrapped `json.loads` in `try/except json.JSONDecodeError`. Failure logs a correlation ID server-side and raises a sanitised `ValueError` with only the reference ID exposed to the UI. |

```python
# BEFORE (v2.1) — vulnerable
raw_json = json.loads(raw_text)

# AFTER (v2.2) — hardened
try:
    raw_json = json.loads(raw_text)
except json.JSONDecodeError as exc:
    err_id = uuid.uuid4().hex[:8].upper()
    log.error(f"[{request_id}] JSON parse failure at pos {exc.pos}: {exc.msg} ({err_id})")
    raise ValueError(f"LLM returned malformed JSON. (Reference: {err_id})") from exc
```

---

#### 🔴 RT-02 — Dict-Bomb via Oversized LLM JSON Response
| Field       | Detail |
|-------------|--------|
| **Severity**   | HIGH |
| **Component**  | `run_agent()` → `AgentReport(**raw_json)` |
| **Vector**     | A prompt-injected or malfunctioning LLM response could return a JSON object with thousands of keys or megabytes of data. `AgentReport(**raw_json)` would unpack the entire dict, causing unbounded memory allocation and CPU usage before Pydantic field validators run. |
| **Impact**     | Memory exhaustion, CPU spike, denial of service |
| **Fix (v2.2)** | Two gates added before Pydantic parse: (1) raw payload byte length must be ≤ 64 KB; (2) top-level key count must be ≤ 20. Both limits are enforced as named constants for easy audit. |

```python
# Gate 1: payload size
if len(raw_text.encode("utf-8")) > MAX_LLM_PAYLOAD_BYTES:   # 65_536
    raise ValueError(f"LLM response exceeded size limit. (Reference: {err_id})")

# Gate 2: key count
if not isinstance(raw_json, dict) or len(raw_json) > MAX_LLM_PAYLOAD_KEYS:  # 20
    raise ValueError(f"LLM response structure invalid. (Reference: {err_id})")
```

---

#### 🟠 RT-03 — No Guard on `response.output` Being `None` or Empty
| Field       | Detail |
|-------------|--------|
| **Severity**   | MEDIUM |
| **Component**  | `run_agent()` → `response.output` iteration |
| **Vector**     | The OpenAI Responses API can return a valid HTTP 200 with a `None` output or empty list under rate limiting, content filtering, or safety refusal conditions. The generator expression `block.text for block in response.output` would raise `TypeError: 'NoneType' is not iterable`. |
| **Impact**     | Unhandled exception, raw traceback potentially visible in logs |
| **Fix (v2.2)** | Explicit guard: `if response is None or not getattr(response, "output", None)` before iteration, with correlation-ID-logged error. |

---

#### 🟠 RT-04 — `float('nan')` Produces Invalid JSON (`NaN`)
| Field       | Detail |
|-------------|--------|
| **Severity**   | MEDIUM |
| **Component**  | `profile_dataframe()` → numeric summary |
| **Vector**     | An all-NaN numeric column (e.g. a column of empty cells cast to float) causes `s.mean()` to return `float('nan')`. Python's `json.dumps` renders this as the bare token `NaN`, which is **not valid per RFC 8259**. The OpenAI API silently rejects or misinterprets such payloads, causing analysis failures or silent bad output. |
| **Impact**     | Silent API failure, incorrect or missing LLM analysis, hard-to-diagnose bug |
| **Reproduction** | Upload a CSV with a fully empty numeric column. |
| **Fix (v2.2)** | `_safe_float()` utility coerces `nan` and `inf` to `None` (JSON `null`) before serialisation. Applied to all six stat fields. |

```python
def _safe_float(v: float) -> float | None:
    """Coerce nan/inf → None to produce spec-compliant JSON null."""
    if math.isnan(v) or math.isinf(v):
        return None
    return v

# Applied at every stat computation:
"mean": _safe_float(round(float(s.mean()), 4)),
```

---

#### 🟠 RT-05 — `pd.util.hash_pandas_object` Is a Private API
| Field       | Detail |
|-------------|--------|
| **Severity**   | MEDIUM |
| **Component**  | Dataset deduplication hash |
| **Vector**     | `pd.util` is an undocumented private namespace. It raises `TypeError` on DataFrames containing mixed or complex dtypes (e.g. `complex128`, custom Extension Arrays) and can produce inconsistent hashes across pandas versions, breaking the session deduplication logic. |
| **Impact**     | Crash on valid dataset upload; silent dedup failures |
| **Fix (v2.2)** | Replaced with `hashlib.sha256` over `df.to_parquet()` bytes — dtype-agnostic, uses the public pandas API, deterministic across versions. Falls back to a random UUID if Parquet serialisation itself fails (exotic dtypes), logging a warning but allowing the session to continue. |

```python
def _stable_dataframe_hash(df: pd.DataFrame) -> str:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    return hashlib.sha256(buf.getvalue()).hexdigest()
```

---

#### 🟡 RT-06 — `except Exception` Too Broad in `sanitize_text`
| Field       | Detail |
|-------------|--------|
| **Severity**   | LOW |
| **Component**  | `sanitize_text()` |
| **Vector**     | `except Exception` catches `MemoryError`, `RecursionError`, and other non-recoverable system exceptions that should propagate. Silently swallowing them returns an empty string, masking serious runtime failures and making debugging extremely difficult. |
| **Impact**     | Silent data loss, masked memory exhaustion bugs |
| **Fix (v2.2)** | Narrowed to `except (TypeError, ValueError, AttributeError)` — the only exceptions that can legitimately arise from `str()` and `html.escape()` on arbitrary input. |

---

#### 🟡 RT-07 — CSS Validator Missed Known Injection Vectors
| Field       | Detail |
|-------------|--------|
| **Severity**   | LOW |
| **Component**  | `_DANGEROUS_CSS` regex |
| **Vector**     | The v2.1 regex `<script\|url\s*\(` did not cover: `expression()` (IE CSS code execution), `@import` (external stylesheet loading), `javascript:` URI scheme in CSS properties, or `-moz-binding:` (Firefox XBL bindings, still seen in legacy pen-test toolkits). A maliciously crafted `style.css` could exploit these vectors to inject code through the `unsafe_allow_html=True` pathway. |
| **Impact**     | CSS-based code execution if style.css is compromised |
| **Fix (v2.2)** | Extended regex pattern covers all six known CSS injection families: |

```python
_DANGEROUS_CSS = re.compile(
    r"<script|url\s*\(|expression\s*\(|@import|javascript\s*:|"
    r"-moz-binding\s*:",
    re.IGNORECASE,
)
```

---

#### 🟡 RT-08 — `uploaded = None` After Size Fail Causes Downstream `AttributeError`
| Field       | Detail |
|-------------|--------|
| **Severity**   | LOW |
| **Component**  | Sidebar file size check |
| **Vector**     | When a file exceeded the 50 MB limit, the code set `uploaded = None` and displayed an error but did **not** halt execution. Streamlit's re-run model means the main body code could execute next, calling `uploaded.name` on `None` and raising an `AttributeError` that leaks a traceback. |
| **Impact**     | Unhandled exception, potential information disclosure |
| **Fix (v2.2)** | `uploaded = None` replaced with `st.stop()` — execution halts immediately after the user-visible error message. |

```python
# BEFORE (v2.1) — vulnerable
uploaded = None  # downstream code may still run

# AFTER (v2.2) — hardened
st.error(f"File exceeds the {MAX_FILE_MB} MB limit.")
st.stop()        # no further execution
```

---

#### 🟡 RT-09 — No Per-Session Rate Limit on LLM Calls
| Field       | Detail |
|-------------|--------|
| **Severity**   | LOW |
| **Component**  | `run_agent()` |
| **Vector**     | With no call budget, a single browser session could trigger unlimited GPT-4.1 API calls (e.g. via automated tooling or a fast-clicking user), leading to uncontrolled API cost amplification with no circuit-breaker. |
| **Impact**     | Unbounded API cost, potential service disruption if rate limits are hit |
| **Fix (v2.2)** | `MAX_REPORTS_PER_SESSION = 5` constant enforced via `st.session_state["llm_call_count"]` counter. Counter incremented only after a successful API round-trip. Remaining budget shown in the sidebar. Button disabled when limit is reached. |

---

### Vulnerability History

| ID | Severity | Version Fixed | Description |
|----|----------|--------------|-------------|
| CRIT-01 | Critical | v2.0 | Raw LLM exec_summary rendered as HTML — XSS via prompt injection chain |
| CRIT-02 | Critical | v2.0 | All metric/insight fields rendered raw — XSS via LLM output |
| HIGH-01 | High | v2.0 | API key stored in `st.session_state` as plaintext |
| HIGH-02 | High | v2.0 | Raw `df.head(5)` rows (PII) sent to OpenAI API |
| HIGH-03 | High | v2.0 | CSS loaded via relative path — path traversal risk |
| HIGH-04 | High | v2.0 | OpenAI API exception detail leaked to UI |
| HIGH-05 | High | v2.0 | File parse exception stack trace leaked to UI |
| MED-01 | Medium | v2.0 | No file size or row count limits |
| MED-02 | Medium | v2.0 | Raw column names sent to LLM — prompt injection via column names |
| MED-03 | Medium | v2.0 | Raw filename interpolated into LLM prompt |
| MED-04 | Medium | v2.0 | LLM JSON output used directly as HTML without validation |
| MED-05 | Medium | v2.0 | No file type validation |
| MED-06 | Medium | v2.0 | No prompt injection guardrails in system prompt |
| LOW-03 | Low | v2.0 | Bare `except Exception: pass` swallowing MemoryError |
| LOW-04 | Low | v2.0 | LLM-supplied quality score used in comparisons with no type check |
| LOW-06 | Low | v2.0 | Dataset filename rendered raw in HTML — XSS via filename |
| RT-01 | High | v2.2 | `json.loads` unguarded — malformed LLM JSON propagates raw exception |
| RT-02 | High | v2.2 | No size/key gate before `AgentReport(**raw_json)` — dict-bomb |
| RT-03 | Medium | v2.2 | `response.output` iteration with no None/empty guard |
| RT-04 | Medium | v2.2 | `float('nan')` → invalid JSON `NaN` sent to OpenAI API |
| RT-05 | Medium | v2.2 | `pd.util.hash_pandas_object` private API crashes on mixed dtypes |
| RT-06 | Low | v2.2 | `except Exception` too broad — masks non-recoverable errors |
| RT-07 | Low | v2.2 | CSS validator missed `expression()`, `@import`, `javascript:` vectors |
| RT-08 | Low | v2.2 | `uploaded = None` allows downstream `AttributeError` on `.name` |
| RT-09 | Low | v2.2 | No per-session LLM call rate limit — cost amplification |

---

### Residual Risk Register

The following risks are accepted with documented rationale and named owners. They are reviewed each release cycle.

| ID | Severity | Description | Mitigation | Owner | Review Date |
|----|----------|-------------|------------|-------|-------------|
| LOW-01 | Low | No SSO / AuthN layer | Network-level access control (VPN / IP allowlist) | Platform/DevOps | Next quarter |
| LOW-05 | Low | f-string HTML interpolation for static strings | All dynamic values wrapped in `sanitize_text()` before interpolation. Code-review checklist item: "No bare variable in HTML f-string." | All developers | Per PR |
| INFO-01 | Info | Downloaded `.txt` report may be ingested by downstream AI systems | AI-generated content header prepended to all exports | Product team | Before PDF/HTML export feature |

---

## Installation

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.11+ |
| pip | 23+ |
| OpenAI API key | GPT-4.1 access required |

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/your-org/enterprise-insight-agent.git
cd enterprise-insight-agent

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set the API key environment variable (never commit this)
export OPENAI_API_KEY="sk-..."   # Linux / macOS
set OPENAI_API_KEY=sk-...        # Windows CMD
$env:OPENAI_API_KEY="sk-..."     # Windows PowerShell

# 5. Run the application
streamlit run autonomous_analysis_agent_v2.2.py
```

### Dependencies

```text
# requirements.txt
streamlit>=1.35.0
pandas>=2.2.0
openpyxl>=3.1.0
pyarrow>=16.0.0
openai>=1.30.0
pydantic>=2.7.0
```

> **Note on pyarrow:** Required for `df.to_parquet()` in the RT-05 dataset hashing fix. If your environment cannot install pyarrow, the application will fall back to UUID-based session tokens (deduplication disabled) with a logged warning — the application remains fully functional.

---

## Configuration

All configuration is via environment variables. No configuration file is read at runtime.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | ✅ Yes | — | OpenAI secret key. Must match `^sk-[A-Za-z0-9\-_]{20,}$` |

### Runtime Constants

The following constants are defined at the top of the source file and can be adjusted before deployment:

| Constant | Default | Description |
|----------|---------|-------------|
| `MAX_FILE_MB` | `50` | Maximum upload size in megabytes |
| `MAX_ROWS` | `500_000` | Maximum rows analysed (dataset is truncated if exceeded) |
| `MAX_REPORTS_PER_SESSION` | `5` | LLM call budget per browser session (RT-09 rate limit) |
| `MAX_LLM_PAYLOAD_KEYS` | `20` | Maximum top-level keys accepted in LLM JSON response (RT-02) |
| `MAX_LLM_PAYLOAD_BYTES` | `65_536` | Maximum byte size of raw LLM JSON response (RT-02) |

---

## Usage

### 1. Upload a Dataset

Use the sidebar file uploader to select a `.csv`, `.xlsx`, or `.xls` file. The application enforces:

- **MIME type** validation server-side against the `ALLOWED_MIMES` allowlist
- **File size** limit of 50 MB (hard stop — does not set `uploaded = None`)
- **Row count** truncation at 500,000 rows after load

### 2. Review the Dataset Preview

Before generating the report, the application displays:

- Row count, column count, missing value count, and memory usage
- An HTML-escaped, 100-row preview of the raw data

### 3. Generate Insights

Click **⚡ Generate Insights**. The button is disabled:

- While a report for the current dataset already exists in the session (spam prevention)
- When the session report limit (default: 5) has been reached

The agent will:

1. Profile the DataFrame (cached — repeated uploads of the same file skip this step)
2. Serialize the sanitized profile to JSON (NaN-safe, no `indent=`)
3. Call GPT-4.1 via the OpenAI Responses API with a hardened system prompt
4. Validate the response through size gates, key count gates, JSON parsing, and Pydantic
5. Render the validated `AgentReport` to the UI

### 4. Download the Report

Use the **⬇ Download Report (.txt)** button to save a plain-text version with an AI-content header.

---

## Data Flow

```
User uploads file
       │
       ▼
MIME check ──fail──▶ st.stop()
       │ pass
       ▼
Size check ──fail──▶ st.stop()
       │ pass
       ▼
pd.read_csv / pd.read_excel
       │
       ▼
Datetime inference (vectorised)
       │
       ▼
Row truncation (if > MAX_ROWS)
       │
       ▼
SHA-256 dataset hash
       │
       ├── hash == cached_hash? ──yes──▶ skip profiling + LLM call
       │                                  render cached report
       ▼ no
profile_dataframe() [@st.cache_data]
  ├── column name sanitization
  ├── NaN-safe float stats (_safe_float)
  ├── categorical value sanitization
  └── sample_rows OMITTED (PII guard)
       │
       ▼
Session rate limit check ──fail──▶ RuntimeError (user-friendly message)
       │ pass
       ▼
OpenAI Responses API (GPT-4.1, timeout=60s)
  └── hardened system prompt with injection guardrails
       │
       ▼
response.output guard (None / empty check)
       │
       ▼
Payload size gate (≤ 64 KB)
       │
       ▼
json.loads (JSONDecodeError guarded)
       │
       ▼
Key count gate (≤ 20 keys)
       │
       ▼
AgentReport(**raw_json) [Pydantic v2]
  ├── all string fields: html.escape() via field validators
  ├── score: clamped 0-100
  ├── list fields: max 20 items, each max 300 chars
  └── sentiment/priority: Literal enum validation
       │
       ▼
st.session_state storage
       │
       ▼
Streamlit rendering (all values pre-escaped)
```

---

## API Reference

### `sanitize_text(value, max_len=2000) → str`

Converts any value to an HTML-safe string. Applies `html.escape()` and truncates. Catches only `(TypeError, ValueError, AttributeError)` — non-recoverable exceptions propagate.

### `sanitize_column_name(name) → str`

Strips column names to `[a-zA-Z0-9_\s]`, max 64 characters. Returns `"unnamed_column"` for empty result.

### `sanitize_filename(name) → str`

Strips filenames to `[a-zA-Z0-9_\-\. ]`, max 128 characters. Returns `"unnamed_file"` for empty result.

### `_safe_float(v) → float | None`

Coerces `float('nan')` and `float('inf')` to `None`. Used in numeric profiling to produce RFC 8259-compliant JSON.

### `_stable_dataframe_hash(df) → str`

Returns a SHA-256 hex digest of the DataFrame via `df.to_parquet()` bytes. Raises on Parquet serialization failure (caller should handle with fallback).

### `profile_dataframe(df) → dict`

`@st.cache_data`-decorated. Builds a sanitized statistical profile. Columns capped at 10 numeric / 8 categorical. No sample rows.

### `run_agent(profile, safe_name) → AgentReport`

Calls GPT-4.1. Enforces session rate limit, response guards, size/key gates, JSON parsing, and Pydantic validation. Raises `ValueError` or `RuntimeError` with correlation IDs on all failure paths.

---

## Performance Characteristics

| Operation | Typical Time | Notes |
|-----------|-------------|-------|
| CSV parse (100 K rows) | ~0.3 s | `encoding_errors="replace"` adds minimal overhead |
| Excel parse (100 K rows) | ~2–5 s | openpyxl is slower than xlrd for large files |
| Datetime inference | ~0.1 s / 1 M cells | Vectorised `df.apply()` single pass |
| `profile_dataframe` | ~0.2 s | Cached on second upload of same file |
| SHA-256 hash (50 MB) | ~0.5 s | Parquet serialization cost amortized by cache benefit |
| GPT-4.1 API call | ~3–8 s | Varies with model load; 60 s timeout enforced |

---

## Development Guide

### Code Review Checklist

Before merging any change to this file, verify:

- [ ] No raw variable interpolated into an HTML f-string without `sanitize_text()` wrapping
- [ ] No new `except Exception:` clauses — narrow to specific exception types
- [ ] No new `json.loads` calls without `json.JSONDecodeError` handling
- [ ] No new `st.session_state` keys storing sensitive values (API keys, PII)
- [ ] Any new LLM output fields added to Pydantic models with appropriate validators
- [ ] `MAX_LLM_PAYLOAD_KEYS` constant updated if schema adds new top-level keys
- [ ] No new file I/O operations without path validation using `BASE_DIR`

### Adding a New LLM Output Field

1. Add the field to the appropriate Pydantic model (`AgentReport`, `InsightItem`, etc.)
2. Add a `@field_validator` with `sanitize_text()` or `escape_list()`
3. Update `MAX_LLM_PAYLOAD_KEYS` if adding a top-level `AgentReport` field
4. Update the `system_prompt` JSON schema block in `run_agent()`
5. Add rendering logic with `unsafe_allow_html=True` only if value is Pydantic-validated

### Running Tests

```bash
# Unit tests (requires pytest)
pytest tests/ -v

# Security-focused linting
pip install bandit
bandit autonomous_analysis_agent_v2.2.py -ll

# Dependency vulnerability scan
pip install safety
safety check -r requirements.txt
```

---

## Deployment

### Streamlit Community Cloud

1. Push the repository to GitHub (ensure `OPENAI_API_KEY` is **not** committed)
2. Connect the repo in [share.streamlit.io](https://share.streamlit.io)
3. Add `OPENAI_API_KEY` under **App Settings → Secrets**
4. Deploy

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY autonomous_analysis_agent_v2.2.py .
# OPTIONAL: COPY style.css .

EXPOSE 8501
CMD ["streamlit", "run", "autonomous_analysis_agent_v2.2.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t insight-agent .
docker run -p 8501:8501 -e OPENAI_API_KEY="sk-..." insight-agent
```

### Production Hardening (Beyond Scope of This File)

For multi-tenant or internet-facing deployments, the following are recommended before go-live. These correspond to residual risks LOW-01 and INFO-01:

- **AuthN layer**: Integrate Okta / Azure AD via Streamlit's `st.experimental_user` or a reverse-proxy with OIDC (e.g. Nginx + oauth2-proxy)
- **Network isolation**: Deploy behind a VPN or IP allowlist; do not expose port 8501 publicly
- **Secrets management**: Use AWS Secrets Manager, GCP Secret Manager, or HashiCorp Vault instead of environment variables in production
- **Monitoring**: Forward structured logs to a SIEM (Splunk, Datadog, CloudWatch) — the correlation IDs in every log line enable end-to-end request tracing
- **Rate limiting**: For public deployments, add an upstream rate-limiting layer (nginx `limit_req`, Cloudflare, AWS WAF) in addition to the in-app session budget

---

## Changelog

### v2.2 — Red Team Hardened *(current)*

- **RT-01 FIXED** `json.loads` now guarded with `json.JSONDecodeError` handler
- **RT-02 FIXED** 64 KB payload size gate and 20-key count gate before Pydantic parse
- **RT-03 FIXED** `response.output` guarded for `None` and empty list
- **RT-04 FIXED** `_safe_float()` coerces `nan`/`inf` → `None` (valid JSON `null`)
- **RT-05 FIXED** `pd.util.hash_pandas_object` replaced with SHA-256 / Parquet hash
- **RT-06 FIXED** `except Exception` narrowed to `(TypeError, ValueError, AttributeError)`
- **RT-07 FIXED** CSS validator extended to cover `expression()`, `@import`, `javascript:`, `-moz-binding:`
- **RT-08 FIXED** `uploaded = None` replaced with `st.stop()` on size failure
- **RT-09 FIXED** Per-session LLM call budget (`MAX_REPORTS_PER_SESSION = 5`) with UI feedback

### v2.1 — Performance & Hygiene

- OpenAI Responses API migration (`client.responses.create`, `gpt-4.1`)
- `timeout=60` on `OpenAI()` client constructor
- `@st.cache_data` on `profile_dataframe()`
- Vectorised datetime inference with `df.apply()`
- `_safe_float` precursor work (NaN identified, fix shipped in v2.2)
- `Field(default_factory=list)` on all Pydantic list fields
- `json.dumps(profile)` without `indent=` (LLM token reduction)
- `memory_usage(index=True)` (faster than `deep=True`)
- Button `disabled` while report in session state
- Dataset hash deduplication (using `pd.util` — replaced in v2.2)
- `ALLOWED_MIMES` enforcement (defined in v2.0, enforced in v2.1)

### v2.0 — Security Remediation

- CRIT-01, CRIT-02: Full HTML-escape pipeline via Pydantic validators
- HIGH-01: API key moved to environment variable
- HIGH-02: Sample rows removed from LLM payload
- HIGH-03: CSS loaded via absolute path with content validation
- HIGH-04, HIGH-05: Exception details suppressed; correlation IDs logged
- MED-01 through MED-06: File limits, column/filename sanitization, injection guardrails

### v1.0 — Initial Build

- Basic Streamlit UI with file upload
- Pandas profiling
- OpenAI `chat.completions` API integration
- JSON report rendering

---

*Enterprise Data Insight Agent — v2.2 Hardened Build*
*Security review completed. Residual risks LOW-01, LOW-05, INFO-01 accepted with documented owners.*
