# Refactor Plan: Native Tool Calling Architecture

**Branch:** `fix/anti-hallucination-guard`  
**Goal:** Stop providers from serializing tool calls as JSON strings. Return typed
`ProviderResponse` objects so `QueryProcessor` never needs to re-parse text to find
tool calls.

---

## Problem Statement

### Current flow (broken)

```
Provider.get_response() → str
  OpenAI:      json.dumps({"tool_calls": [...]})      ← if tool call
               "plain text"                            ← if no tool call
  Gemini:      json.dumps({"functionCall": {...}})     ← if tool call
               first_part["text"]                      ← if no tool call
  Anthropic:   json.dumps({"tool_use": {...}, ...})    ← if tool call
               " ".join(text_parts)                    ← if no tool call

QueryProcessor._detect_function_call(response_text)
  → ResponseParser.parse(response_text)   # tries to JSON-decode the string
  → FunctionCallParser.detect()           # tries openai / gemini / anthropic shapes
  → list[FunctionCall] | None
```

**Why this causes hallucinations:**

1. **False negatives (the main danger):** A model that skips tool calling and writes
   prose like *"I would call `turn_on` with entity_id='light.living_room'"* returns
   plain text. `FunctionCallParser` gets `type=text` from `ResponseParser` and returns
   `None`. QueryProcessor then shows the prose to the user as if it were a valid
   response. The hallucination guard catches some of these, but it's a band-aid.

2. **False positives (rare but possible):** Any text that parses as JSON and happens
   to contain `functionCall`, `tool_use`, or `tool_calls` keys will trigger a tool
   call. A model explaining its previous actions could accidentally trigger this.

3. **Round-trip encoding fragility:** `json.dumps → json.loads → format-detect` is
   three layers of string manipulation. Each adds potential breakage (invisible chars,
   encoding issues, key collision).

### Streaming is already partially fixed

The streaming path already returns structured chunks:
```python
# Anthropic/AnthropicOAuth/GeminiOAuth streaming already yields:
{"type": "tool_call", "name": str, "args": dict, "id": str}
{"type": "text", "content": str}
```

`QueryProcessor.process_stream()` uses these directly without text parsing. **Only
the non-streaming `get_response()` path is broken.**

### OpenClaw's approach (reference)

OpenClaw (`pi-tool-definition-adapter.ts`) wraps all tools in typed `ToolDefinition`
objects. The agent core returns `AgentEvent` typed objects for tool start/update/end,
never serialized strings. Tool calls arrive as structured events, not embedded text.
The key insight: **tool call detection happens at the SDK level, not by string parsing
at the application level.**

---

## Phase 0: New Types (no breaking changes)

**File to create:** `custom_components/homeclaw/providers/types.py`

```python
"""Provider-agnostic response types for structured tool call passing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProviderToolCall:
    """A single tool call returned by a provider.

    Attributes:
        id:        Unique call ID (from provider; generated if absent).
        name:      Tool/function name.
        arguments: Parsed argument dictionary (never a string).
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderResponse:
    """Structured response from an AI provider.

    Replaces the current pattern of returning str that may contain
    JSON-encoded tool calls. Tool calls are always in `tool_calls`;
    plain text goes in `text`.

    Attributes:
        text:         Model text (may be non-empty even when tool_calls present,
                      e.g. Anthropic sending reasoning before a tool call).
        tool_calls:   Zero or more tool calls. Empty list = no tool call.
        raw_response: The raw API response dict for debugging/logging.
    """

    text: str = ""
    tool_calls: list[ProviderToolCall] = field(default_factory=list)
    raw_response: dict[str, Any] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        """True if the response contains at least one tool call."""
        return bool(self.tool_calls)
```

**Why a new file?** `registry.py` already has `AIProvider` and `ProviderRegistry`.
Putting types there would create circular imports (providers import `registry.py`).
A separate `types.py` is clean and importable by all providers without cycles.

**No breaking changes at this step.** Nothing uses `ProviderResponse` yet.

---

## Phase 1: Provider Refactor (each provider separately)

### Strategy

1. Change `AIProvider.get_response()` abstract method signature in `registry.py` to
   return `ProviderResponse | str` (union) during the migration window.
2. Add `supports_native_response: bool = False` property to `AIProvider` so
   `QueryProcessor` knows which return type to expect.
3. Migrate providers one by one: change `_extract_response()` to return
   `ProviderResponse`, set `supports_native_response = True`.
4. `QueryProcessor` checks `supports_native_response` and uses the typed path.
5. Once all providers are migrated, remove the union type and `supports_native_response`.

---

### 1.1 registry.py — Add transition support

**File:** `custom_components/homeclaw/providers/registry.py`  
**Current `AIProvider.get_response()` signature (line 35–44):**

```python
@abstractmethod
async def get_response(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
    """Get a response from the AI provider.
    ...
    Returns:
        The AI response as a string.
    """
```

**Proposed change — add `supports_native_response` property:**

```python
from .types import ProviderResponse  # NEW

class AIProvider(ABC):
    ...

    @property
    def supports_native_response(self) -> bool:
        """Return True if get_response() returns ProviderResponse objects.

        Providers should override this to True after they implement the
        native ProviderResponse return type. QueryProcessor uses this flag
        to skip FunctionCallParser text scanning.
        """
        return False  # Safe default: old string-based behaviour

    @abstractmethod
    async def get_response(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> "ProviderResponse | str":
        """Get a response from the AI provider.
        ...
        Returns:
            ProviderResponse when supports_native_response is True,
            str otherwise (legacy).
        """
```

**No other changes to `registry.py` yet.**

---

### 1.2 base_client.py — Update `_extract_response` abstract method

**File:** `custom_components/homeclaw/providers/base_client.py`  
**Current `_extract_response` signature (line 61–70):**

```python
@abstractmethod
def _extract_response(self, response_data: dict[str, Any]) -> str:
    """Extract the response text from the API response.
    ...
    Returns:
        The extracted response text.
    """
```

**Proposed change:**

```python
from .types import ProviderResponse  # NEW

@abstractmethod
def _extract_response(
    self, response_data: dict[str, Any]
) -> "ProviderResponse | str":
    """Extract the response from the API response.

    Returns either a legacy str (providers not yet migrated) or a
    ProviderResponse (migrated providers).
    """
```

**`BaseHTTPClient.get_response()` (line 77–130) — only 2-line change:**

```python
async def get_response(
    self, messages: list[dict[str, Any]], **kwargs: Any
) -> "ProviderResponse | str":
    ...
    # (retry loop unchanged)
    ...
    if response.status == 200:
        response_data = await response.json()
        return self._extract_response(response_data)  # return type widens
    ...
```

No logic change needed — the template method already calls `_extract_response()`.

---

### 1.3 OpenAI provider

**File:** `custom_components/homeclaw/providers/openai.py`  
**Lines affected:** 162–188 (`_extract_response`), plus new property

**Current `_extract_response` (lines 162–188):**

```python
def _extract_response(self, response_data: dict[str, Any]) -> str:
    choices = response_data.get("choices", [])
    if not choices:
        return ""

    message = choices[0].get("message", {})

    # Check for tool calls
    tool_calls = message.get("tool_calls")
    if tool_calls:
        return json.dumps({"tool_calls": tool_calls})  # ← problem line

    # Return regular content
    content = message.get("content")
    return content if content is not None else ""
```

**Proposed `_extract_response`:**

```python
from .types import ProviderResponse, ProviderToolCall  # NEW

@property
def supports_native_response(self) -> bool:
    return True  # NEW

def _extract_response(self, response_data: dict[str, Any]) -> ProviderResponse:
    choices = response_data.get("choices", [])
    if not choices:
        return ProviderResponse(raw_response=response_data)

    message = choices[0].get("message", {})

    # Native tool calls from OpenAI API
    raw_tool_calls = message.get("tool_calls")
    if raw_tool_calls:
        tool_calls = []
        for tc in raw_tool_calls:
            func = tc.get("function", {})
            raw_args = func.get("arguments", {})
            # OpenAI returns arguments as a JSON string — parse it
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args)
                except (json.JSONDecodeError, TypeError):
                    args = {}
            else:
                args = raw_args if isinstance(raw_args, dict) else {}
            tool_calls.append(ProviderToolCall(
                id=tc.get("id", func.get("name", "")),
                name=func.get("name", ""),
                arguments=args,
            ))
        return ProviderResponse(
            text=message.get("content") or "",
            tool_calls=tool_calls,
            raw_response=response_data,
        )

    content = message.get("content")
    return ProviderResponse(
        text=content if content is not None else "",
        raw_response=response_data,
    )
```

**Remove import:** `from ..core.tool_call_codec import extract_tool_calls_from_assistant_content`
is still needed for `_convert_multimodal_messages` — do NOT remove yet. That method
reconstructs history messages and is a separate concern.

**Provider-specific quirk:** OpenAI returns `arguments` as a serialized JSON string
inside `function.arguments`. Must parse it in `_extract_response`.

---

### 1.4 Groq provider

**File:** `custom_components/homeclaw/providers/groq.py`  
**Lines affected:** 1–32 (entire file)

Groq inherits everything from `OpenAIProvider` and only overrides `api_url`. Since
`OpenAIProvider._extract_response` will return `ProviderResponse` after 1.3, **Groq
gets native responses for free.**

Only change needed:

```python
# groq.py — no code change needed after openai.py is migrated
# supports_native_response is inherited from OpenAIProvider → True
```

**Zero lines to change in groq.py.** Verify by adding a test.

---

### 1.5 OpenRouter provider

**File:** `custom_components/homeclaw/providers/openrouter.py`  
**Lines affected:** 87–108 (`_extract_response`)

**Current `_extract_response` (lines 87–108) — identical to OpenAI pattern:**

```python
def _extract_response(self, response_data: dict[str, Any]) -> str:
    choices = response_data.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    tool_calls = message.get("tool_calls")
    if tool_calls:
        return json.dumps({"tool_calls": tool_calls})  # ← problem line
    content = message.get("content")
    return content if content is not None else ""
```

**Proposed `_extract_response`:**

```python
from .types import ProviderResponse, ProviderToolCall  # NEW

@property
def supports_native_response(self) -> bool:
    return True  # NEW

def _extract_response(self, response_data: dict[str, Any]) -> ProviderResponse:
    choices = response_data.get("choices", [])
    if not choices:
        return ProviderResponse(raw_response=response_data)

    message = choices[0].get("message", {})
    raw_tool_calls = message.get("tool_calls")

    if raw_tool_calls:
        tool_calls = []
        for tc in raw_tool_calls:
            func = tc.get("function", {})
            raw_args = func.get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args)
                except (json.JSONDecodeError, TypeError):
                    args = {}
            else:
                args = raw_args if isinstance(raw_args, dict) else {}
            tool_calls.append(ProviderToolCall(
                id=tc.get("id", func.get("name", "")),
                name=func.get("name", ""),
                arguments=args,
            ))
        return ProviderResponse(
            text=message.get("content") or "",
            tool_calls=tool_calls,
            raw_response=response_data,
        )

    content = message.get("content")
    return ProviderResponse(
        text=content if content is not None else "",
        raw_response=response_data,
    )
```

**Note:** Consider extracting this OpenAI-compatible extraction into a shared helper
`_openai_compatible_extract(response_data) -> ProviderResponse` in `openai.py` and
calling it from both `OpenAIProvider` and `OpenRouterProvider` to avoid duplication.

---

### 1.6 Gemini provider (API key)

**File:** `custom_components/homeclaw/providers/gemini.py`  
**Lines affected:** 139–178 (`_extract_response`), 198–255 (`get_response` override)

**Current `_extract_response` (lines 139–178):**

```python
def _extract_response(self, response_data: dict[str, Any]) -> str:
    import json
    candidates = response_data.get("candidates", [])
    if not candidates:
        raise ValueError("No response from Gemini API (empty candidates)")
    candidate = candidates[0]
    content = candidate.get("content", {})
    parts = content.get("parts", [])
    if parts:
        first_part = parts[0]
        if "functionCall" in first_part:
            return json.dumps(first_part)          # ← problem: entire part as JSON
        if "text" in first_part:
            return first_part["text"]
        return ""
    return ""
```

**Provider-specific quirk:** Gemini may include `thoughtSignature` alongside
`functionCall` in the part. The current code preserves the entire `first_part` dict
via `json.dumps(first_part)` so that later when building history, the signature is
preserved. **The refactored version must keep this in `raw_response`.**

Also: Gemini may have multiple parts. The current code only looks at `parts[0]`.
Check if multi-part responses need handling (they do in streaming but the non-streaming
path has always only checked `parts[0]`).

**Proposed `_extract_response`:**

```python
from .types import ProviderResponse, ProviderToolCall  # NEW

@property
def supports_native_response(self) -> bool:
    return True  # NEW

def _extract_response(self, response_data: dict[str, Any]) -> ProviderResponse:
    import json
    candidates = response_data.get("candidates", [])
    if not candidates:
        raise ValueError("No response from Gemini API (empty candidates)")

    candidate = candidates[0]
    content = candidate.get("content", {})
    parts = content.get("parts", [])

    if not parts:
        return ProviderResponse(raw_response=response_data)

    first_part = parts[0]

    if "functionCall" in first_part:
        fc = first_part["functionCall"]
        tool_call = ProviderToolCall(
            id=f"gemini_{fc.get('name', '')}",
            name=fc.get("name", ""),
            arguments=fc.get("args", {}) if isinstance(fc.get("args"), dict) else {},
        )
        return ProviderResponse(
            text="",
            tool_calls=[tool_call],
            # Preserve the FULL part dict (including thoughtSignature) in raw_response.
            # QueryProcessor uses this when building assistant history messages.
            raw_response={"_gemini_raw_part": first_part, **response_data},
        )

    if "text" in first_part:
        return ProviderResponse(
            text=first_part["text"],
            raw_response=response_data,
        )

    return ProviderResponse(raw_response=response_data)
```

**Critical: `thoughtSignature` preservation.** Today `process_stream()` in
`QueryProcessor` (lines ~770–800) already handles `_raw_function_call` from streaming:

```python
tool_call_obj = accumulated_tool_calls[0].get("_raw_function_call")
if not tool_call_obj:
    assistant_tool_json = build_assistant_tool_message(normalized_tool_calls)
    built_messages.append({"role": "assistant", "content": assistant_tool_json})
else:
    built_messages.append({"role": "assistant", "content": json.dumps(tool_call_obj)})
```

For non-streaming, `QueryProcessor.process()` must do the same: when
`response.raw_response["_gemini_raw_part"]` exists, use it as the history message
content. This is documented further in Phase 2.

---

### 1.7 GeminiOAuth provider

**File:** `custom_components/homeclaw/providers/gemini_oauth.py`  
**Lines affected:** `_do_request()` method (lines 438–558), `get_response()` (lines 560–633)

**Current response extraction in `_do_request()` (lines 519–558):**

```python
if "functionCall" in first_part:
    func_name = first_part["functionCall"].get("name", "unknown")
    _LOGGER.debug("Gemini OAuth function call detected: %s", func_name)
    return json.dumps(first_part)                # ← problem line
if "text" in first_part:
    text_response = first_part["text"]
    return text_response
# ...fallbacks...
return json.dumps(data)
```

**`_do_request()` currently returns `str`. Change to return `ProviderResponse`.**

This method is called by `get_response()` which also returns `str`.

**Proposed `_do_request()` return type change:**

```python
from .types import ProviderResponse, ProviderToolCall  # NEW

# _do_request becomes:
async def _do_request(
    self,
    session: aiohttp.ClientSession,
    url: str,
    headers: dict,
    wrapped_payload: dict,
) -> ProviderResponse:  # Changed from str
    ...
    # In the extraction section:
    if "functionCall" in first_part:
        fc = first_part["functionCall"]
        func_name = fc.get("name", "unknown")
        _LOGGER.debug("Gemini OAuth function call detected: %s", func_name)
        tool_call = ProviderToolCall(
            id=f"gemini_{func_name}",
            name=func_name,
            arguments=fc.get("args", {}) if isinstance(fc.get("args"), dict) else {},
        )
        return ProviderResponse(
            text="",
            tool_calls=[tool_call],
            # Preserve full part for thoughtSignature
            raw_response={"_gemini_raw_part": first_part, **data},
        )

    if "text" in first_part:
        return ProviderResponse(
            text=first_part["text"],
            raw_response=data,
        )

    _LOGGER.warning("Unexpected Gemini OAuth response format: %s", response_text[:500])
    return ProviderResponse(raw_response=data)
```

**`get_response()` updated signature:**

```python
@property
def supports_native_response(self) -> bool:
    return True  # NEW

async def get_response(self, messages, **kwargs) -> ProviderResponse:
    ...
    return await self._retry_with_backoff(
        self._do_request, session, url, headers, wrapped_payload
    )
    # return type propagates from _do_request
```

**`_retry_with_backoff` also must propagate `ProviderResponse`** — its return type
annotation changes from implicit `Any` to `ProviderResponse`. The method is generic
enough that no body changes are needed, just the annotation.

**Streaming unchanged**: `get_response_stream()` already yields structured chunks
and is not affected by this refactor.

---

### 1.8 Anthropic provider (API key)

**File:** `custom_components/homeclaw/providers/anthropic.py`  
**Lines affected:** 252–303 (`_extract_response`), `_extract_response` also returns
the `json.dumps(result)` blob currently.

**Current `_extract_response` (lines 252–303):**

```python
def _extract_response(self, response_data: dict[str, Any]) -> str:
    content_blocks = response_data.get("content", [])
    if not content_blocks:
        return ""

    text_parts = []
    tool_use_blocks = []
    for block in content_blocks:
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_use_blocks.append(block)

    if tool_use_blocks:
        result: dict[str, Any] = {
            "tool_use": {
                "id": tool_use_blocks[0].get("id"),
                "name": tool_use_blocks[0].get("name"),
                "input": tool_use_blocks[0].get("input"),
            }
        }
        if text_parts:
            result["text"] = " ".join(text_parts)
        if len(tool_use_blocks) > 1:
            result["additional_tool_calls"] = [...]
        return json.dumps(result)            # ← problem line

    return " ".join(text_parts) if text_parts else ""
```

**Provider-specific quirk:** Anthropic can return MULTIPLE `tool_use` blocks in one
response (parallel tool calling). The current code handles this via `additional_tool_calls`.
The new code must return all of them as `ProviderToolCall` objects.

**Also:** Anthropic can return pre-tool-call text (thinking text before calling a
tool). Currently this is embedded in `result["text"]`. The new `ProviderResponse.text`
field handles this naturally.

**Proposed `_extract_response`:**

```python
from .types import ProviderResponse, ProviderToolCall  # NEW

@property
def supports_native_response(self) -> bool:
    return True  # NEW

def _extract_response(self, response_data: dict[str, Any]) -> ProviderResponse:
    content_blocks = response_data.get("content", [])
    if not content_blocks:
        return ProviderResponse(raw_response=response_data)

    text_parts: list[str] = []
    tool_calls: list[ProviderToolCall] = []

    for block in content_blocks:
        block_type = block.get("type")
        if block_type == "text":
            text_parts.append(block.get("text", ""))
        elif block_type == "tool_use":
            tool_calls.append(ProviderToolCall(
                id=block.get("id", ""),
                name=block.get("name", ""),
                arguments=block.get("input", {}) if isinstance(block.get("input"), dict) else {},
            ))

    return ProviderResponse(
        text=" ".join(text_parts) if text_parts else "",
        tool_calls=tool_calls,
        raw_response=response_data,
    )
```

**After this change:** The serialization round-trip
`json.dumps({"tool_use": {...}}) → json.loads → FunctionCallParser._try_anthropic()`
is completely eliminated for non-streaming requests.

**`get_response_stream()` unchanged** — already yields structured chunks.

---

### 1.9 AnthropicOAuth provider

**File:** `custom_components/homeclaw/providers/anthropic_oauth.py`  
**Lines affected:** `get_response()` method (lines 218–405), specifically
the extraction section at lines 364–405.

AnthropicOAuth does **not** use `BaseHTTPClient` — it overrides `get_response()`
directly (uses its own `aiohttp.ClientSession`). The extraction is inline.

**Current extraction in `get_response()` (lines 364–405):**

```python
content_blocks = data.get("content", [])
if not content_blocks:
    return ""

text_parts = []
tool_use_blocks = []
for block in content_blocks:
    if block.get("type") == "text":
        text_parts.append(block.get("text", ""))
    elif block.get("type") == "tool_use":
        tool_use_blocks.append(block)

if tool_use_blocks:
    result: dict[str, Any] = {
        "tool_use": {
            "id": tool_use_blocks[0].get("id"),
            "name": tool_use_blocks[0].get("name"),
            "input": tool_use_blocks[0].get("input"),
        }
    }
    if text_parts:
        result["text"] = " ".join(text_parts)
    if len(tool_use_blocks) > 1:
        result["additional_tool_calls"] = [...]
    return json.dumps(result)                # ← problem line

return " ".join(text_parts) if text_parts else ""
```

**Proposed extraction (replace those lines in `get_response()`):**

```python
from .types import ProviderResponse, ProviderToolCall  # NEW (top of file)

# ...inside get_response(), after `data = json.loads(response_text)`:

content_blocks = data.get("content", [])
if not content_blocks:
    return ProviderResponse(raw_response=data)

text_parts: list[str] = []
tool_calls: list[ProviderToolCall] = []

for block in content_blocks:
    block_type = block.get("type")
    if block_type == "text":
        text_parts.append(block.get("text", ""))
    elif block_type == "tool_use":
        tool_calls.append(ProviderToolCall(
            id=block.get("id", ""),
            name=block.get("name", ""),
            arguments=block.get("input", {}) if isinstance(block.get("input"), dict) else {},
        ))

return ProviderResponse(
    text=" ".join(text_parts) if text_parts else "",
    tool_calls=tool_calls,
    raw_response=data,
)
```

**Add property:**

```python
@property
def supports_native_response(self) -> bool:
    return True
```

**`get_response()` signature:**

```python
async def get_response(self, messages: list[dict[str, Any]], **kwargs: Any) -> ProviderResponse:
```

**`get_response_stream()` unchanged.**

---

### 1.10 Local provider

**File:** `custom_components/homeclaw/providers/local.py`  
**Lines affected:** 76–86 (`_extract_response`)

LocalProvider has `supports_tools: False` by default. It never emits tool calls.

**No migration urgency.** However, for consistency, wrap plain text in `ProviderResponse`:

```python
from .types import ProviderResponse  # NEW

@property
def supports_native_response(self) -> bool:
    return True  # NEW

def _extract_response(self, response_data: dict[str, Any]) -> ProviderResponse:
    text = response_data.get("message", {}).get("content", "")
    return ProviderResponse(text=text, raw_response=response_data)
```

If a local model is configured with `supports_tools: True`, the model may return
OpenAI-compatible `tool_calls` in its response. For now, those are handled by the
`FunctionCallParser` fallback (see Phase 2). Extend later if a local model confirms
native tool call support.

---

## Phase 2: QueryProcessor Refactor

**File:** `custom_components/homeclaw/core/query_processor.py`

### 2.1 `process()` method (lines 936–1157)

**Current pattern (lines ~1039–1090):**

```python
response_text = await self.provider.get_response(
    built_messages, **provider_kwargs
)

# Detect function call
function_calls = self._detect_function_call(response_text, allowed_tool_names=allowed_names_p)

if not function_calls:
    # hallucination check, return response_text as text
    ...
    return {"success": True, "response": response_text, "messages": updated_messages}

# Handle function calls
built_messages.append({"role": "assistant", "content": response_text})
```

**Proposed pattern:**

```python
raw_response = await self.provider.get_response(
    built_messages, **provider_kwargs
)

# Unwrap to ProviderResponse
if isinstance(raw_response, str):
    # Legacy provider: run through text parser (FunctionCallParser fallback)
    provider_resp = self._legacy_string_to_response(raw_response, allowed_names_p)
else:
    provider_resp = raw_response  # Already a ProviderResponse
    # Still validate tool names
    if provider_resp.tool_calls:
        valid_calls = [tc for tc in provider_resp.tool_calls if tc.name in allowed_names_p]
        invalid = [tc.name for tc in provider_resp.tool_calls if tc.name not in allowed_names_p]
        if invalid:
            _LOGGER.warning("Rejected hallucinated tool calls from native response: %s", invalid)
        provider_resp = ProviderResponse(
            text=provider_resp.text,
            tool_calls=valid_calls,
            raw_response=provider_resp.raw_response,
        )

if not provider_resp.has_tool_calls:
    # Hallucination check (only on text path — native tool calls can't be hallucinations)
    hal_result = detect_hallucinated_actions(provider_resp.text, tool_calls_made=[])
    ...
    return {
        "success": True,
        "response": provider_resp.text,
        "messages": updated_messages,
    }

# Has tool calls — build history message
response_for_history = self._build_history_message(provider_resp)
built_messages.append({"role": "assistant", "content": response_for_history})

# Convert ProviderToolCall → FunctionCall
function_calls = [
    FunctionCall(
        id=tc.id,
        name=tc.name,
        arguments=tc.arguments,
    )
    for tc in provider_resp.tool_calls
]
```

**New helper methods to add to `QueryProcessor`:**

```python
def _legacy_string_to_response(
    self,
    response_text: str,
    allowed_names: set[str],
) -> ProviderResponse:
    """Wrap a legacy string response in ProviderResponse using FunctionCallParser."""
    from ..providers.types import ProviderResponse, ProviderToolCall
    
    function_calls = self._detect_function_call(response_text, allowed_tool_names=allowed_names)
    if not function_calls:
        return ProviderResponse(text=response_text)
    
    # Extract any pre-tool text (Anthropic embeds it as "text" key)
    try:
        parsed = json.loads(response_text)
        pre_text = parsed.get("text", "") if isinstance(parsed, dict) else ""
    except (json.JSONDecodeError, ValueError):
        pre_text = ""
    
    return ProviderResponse(
        text=pre_text,
        tool_calls=[
            ProviderToolCall(id=fc.id, name=fc.name, arguments=fc.arguments)
            for fc in function_calls
        ],
    )

def _build_history_message(self, provider_resp: "ProviderResponse") -> str:
    """Build the assistant history message string from a ProviderResponse.
    
    For Gemini with thoughtSignature: preserve the raw part.
    For all others: use build_assistant_tool_message().
    """
    from ..providers.types import ProviderResponse
    
    # Gemini thoughtSignature preservation
    raw_part = provider_resp.raw_response.get("_gemini_raw_part")
    if raw_part:
        return json.dumps(raw_part)
    
    # Standard canonical format
    normalized = [
        {"id": tc.id, "name": tc.name, "args": tc.arguments}
        for tc in provider_resp.tool_calls
    ]
    return build_assistant_tool_message(normalized)
```

### 2.2 `process_stream()` method — non-streaming fallback path (lines ~618–680)

The non-streaming fallback branch (when provider doesn't support streaming) has the
same `response_text = await self.provider.get_response(...)` + `_detect_function_call`
pattern. Apply the same `_legacy_string_to_response` / `ProviderResponse` handling:

```python
# Lines ~618–643 in process_stream() (non-streaming fallback)
raw_response = await self.provider.get_response(
    built_messages, **provider_kwargs
)
provider_resp = (
    raw_response if not isinstance(raw_response, str)
    else self._legacy_string_to_response(raw_response, allowed_names)
)

if provider_resp.has_tool_calls:
    if provider_resp.text:
        yield TextEvent(content=provider_resp.text)
    # ... process tool calls
else:
    yield TextEvent(content=provider_resp.text)
```

### 2.3 `process_stream()` — streaming path (lines ~700+)

**The streaming path already works correctly.** Providers already yield structured
chunks. No changes needed here.

The streaming path accumulates:
- `{"type": "text", "content": str}` → `accumulated_text`
- `{"type": "tool_call", "name": str, "args": dict, "id": str}` → `accumulated_tool_calls`

This is already the equivalent of `ProviderResponse`. No refactor needed.

### 2.4 `_repair_tool_history()` method

This method (lines ~225–285) also calls `self._detect_function_call(content)` when
scanning history for pending tool calls. This is necessary because history messages
are stored as JSON strings (the `content` field of `role: assistant` messages).

**This call stays:** history scanning will continue to use text parsing because
history messages must be stored as strings (JSON format) regardless of provider.
The fix in Phase 2 only affects the **live response** from the provider, not stored
history.

However, after Phase 1, we can also clean up history encoding. Long-term, history
could store `ProviderResponse`-equivalent dicts rather than JSON strings. That is
out of scope for this refactor.

### 2.5 `_detect_function_call()` method status

After Phase 1 + 2 are complete for all providers:
- `_detect_function_call()` is only called from:
  - `_legacy_string_to_response()` — fallback for non-migrated providers
  - `_repair_tool_history()` — history scanning (always needed)
- It is NOT called on live provider responses for migrated providers

**Keep `_detect_function_call()`.** It's still needed for:
1. Legacy local models that return OpenAI-format tool calls as text
2. History repair scanning
3. Any future non-standard provider

### 2.6 What happens to `FunctionCallParser`

**Keep but scope it to legacy path only.**

After migration:
- `FunctionCallParser.detect()` is only called via `_legacy_string_to_response()`
- And `_repair_tool_history()` for history scanning

**Do NOT delete `function_call_parser.py`.** It's needed for:
- Local models that may format tool calls as JSON text
- Backward compatibility with any custom providers
- History repair

**Consider renaming** to `LegacyFunctionCallParser` or `TextFunctionCallParser` to
signal its reduced scope. Add a module docstring note: *"This parser is a legacy
fallback for providers that serialize tool calls as JSON strings. Native providers
should return ProviderResponse objects instead."*

### 2.7 What happens to `ResponseParser`

`ResponseParser` is used only by `FunctionCallParser`. Since `FunctionCallParser`
is kept as a legacy fallback, `ResponseParser` stays too.

No changes needed to `response_parser.py`.

### 2.8 What happens to `tool_call_codec.py`

`tool_call_codec.py` provides:
- `normalize_tool_calls()` — normalizes streaming tool call dicts
- `build_assistant_tool_message()` — builds history message JSON
- `extract_tool_calls_from_assistant_content()` — decodes history messages

**All three functions remain necessary:**
- `normalize_tool_calls()`: still used in `process_stream()` for streaming chunks
- `build_assistant_tool_message()`: called from `_build_history_message()` (new helper)
- `extract_tool_calls_from_assistant_content()`: used by Anthropic/AnthropicOAuth
  when reconstructing history from stored JSON

**No deletions from `tool_call_codec.py`.**

---

## Phase 3: Streaming Refactor

### Current streaming format (already structured)

All three streaming providers (Anthropic, AnthropicOAuth, GeminiOAuth) already
yield structured dicts. Non-streaming providers (OpenAI, Gemini, OpenRouter, Groq,
Local) fall back to `BaseHTTPClient.get_response_stream()` which wraps the text
response in `{"type": "text", "content": text}`.

**Streaming already works correctly.** The `process_stream()` accumulation logic
correctly separates text from tool calls.

### Minor streaming improvement: type consistency

The streaming chunks use `"args"` for arguments:

```python
{"type": "tool_call", "name": str, "args": dict, "id": str}
```

But `FunctionCallParser` in the non-streaming path ultimately produces `FunctionCall`
with `.arguments` (not `.args`). There's a translation step in `process_stream()`:

```python
function_calls = [
    FunctionCall(
        id=tc.get("id") or tc.get("name", "unknown"),
        name=tc["name"],
        arguments=tc.get("args", {}),        # "args" → .arguments
    )
    for tc in normalized_tool_calls
]
```

This is acceptable. No change needed; the convention is consistent within its layer.

### No streaming providers to migrate

The streaming path is already clean. Phase 3 is essentially a no-op.

---

## Phase 4: Cleanup

### After all providers migrated and tests passing:

#### Files to simplify (not delete):

| File | Change |
|------|--------|
| `core/function_call_parser.py` | Add deprecation note; rename module docstring |
| `core/response_parser.py` | Add deprecation note; will only serve legacy path |
| `providers/registry.py` | Remove `supports_native_response` property fallback (make abstract) |
| `providers/base_client.py` | Change `_extract_response` return type from union to `ProviderResponse` only |

#### Remove the `supports_native_response` flag:

Once ALL providers return `ProviderResponse`, remove:
1. `AIProvider.supports_native_response` property from `registry.py`
2. `isinstance(raw_response, str)` checks from `QueryProcessor`
3. `_legacy_string_to_response()` helper (or keep as a no-op stub for custom providers)

#### Tests to update:

**`tests/test_providers/test_openai.py`**
- All assertions on `_extract_response()` return type change from `str` to `ProviderResponse`
- Pattern: `assert result == json.dumps(...)` → `assert result.tool_calls[0].name == "..."`

**`tests/test_providers/test_groq.py`**
- Same pattern — inherits from OpenAI test patterns

**`tests/test_providers/test_openrouter.py`**
- Same pattern

**`tests/test_providers/test_gemini.py`**
- `assert result == json.dumps({"functionCall": ...})` → `assert result.tool_calls[0].name == ...`
- Verify `raw_response["_gemini_raw_part"]` for thoughtSignature preservation

**`tests/test_providers/test_gemini_oauth.py`**
- `_do_request()` return type assertions

**`tests/test_providers/test_anthropic.py`**
- Multi-tool-use assertions: `result.tool_calls` list, not `additional_tool_calls` JSON

**`tests/test_providers/test_anthropic_oauth.py`**
- Same as anthropic

**`tests/test_providers/test_local.py`**
- `result.text == "..."` instead of `result == "..."`

**`tests/test_core/test_query_processor.py`**
- Mock provider's `get_response()` to return `ProviderResponse` instead of str
- Remove test cases that rely on string parsing of tool calls

**`tests/test_core/test_function_call_parser.py`**
- No changes needed (tests the legacy path which is kept)

**`tests/test_core/test_response_parser.py`**
- No changes needed

**`tests/test_core/test_tool_call_codec.py`**
- No changes needed

---

## Migration Strategy

### Can this be done incrementally? YES.

The `supports_native_response` flag enables incremental migration:

```
Week 1: Phase 0 (add types.py — zero risk, zero tests needed)
Week 1: Phase 1.3 — Migrate OpenAI (most-tested provider)
Week 1: Phase 1.4 — Groq (free, inherits from OpenAI)
Week 2: Phase 1.5 — OpenRouter (copy-paste from OpenAI)
Week 2: Phase 1.8 — Anthropic (parallel tool use, well-tested)
Week 2: Phase 1.9 — AnthropicOAuth
Week 3: Phase 1.6 — Gemini (thoughtSignature quirk)
Week 3: Phase 1.7 — GeminiOAuth
Week 3: Phase 1.10 — Local (trivial)
Week 4: Phase 2 — QueryProcessor + tests
Week 5: Phase 4 — Cleanup
```

**Backward compatibility during transition:**

The `isinstance(raw_response, str)` check in `QueryProcessor` ensures that:
- Migrated providers: use `ProviderResponse` directly (no string parsing)
- Non-migrated providers: fall back to `_legacy_string_to_response()` → `FunctionCallParser`
- Custom providers: if they return `str`, the legacy path handles them

### Feature flag approach

The `supports_native_response` property IS the feature flag. No config needed.

---

## Risk Assessment

### Risk 1: thoughtSignature breaking Gemini history (HIGH)

**Problem:** Gemini API docs require the EXACT `functionCall` part to be echoed back
in history, including `thoughtSignature`. The current code does `json.dumps(first_part)`
to preserve the entire part dict. The new `ProviderResponse.raw_response["_gemini_raw_part"]`
approach must be used consistently everywhere `process_stream()` and `process()` build
assistant history messages.

**How to catch:** Integration test: make a Gemini function call → verify next request
sends the correct history → verify Gemini doesn't complain about missing thoughtSignature.

**Rollback:** Revert `gemini.py` and `gemini_oauth.py` changes; restore `json.dumps(first_part)`.

### Risk 2: AnthropicOAuth multi-tool parallel calls (MEDIUM)

**Problem:** Anthropic supports parallel tool calling (multiple `tool_use` blocks
in one response). The current code handles `additional_tool_calls`. The new
`ProviderResponse.tool_calls` list handles multiple calls natively. Verify that
`process()` and `process_stream()` in `QueryProcessor` correctly execute ALL tool
calls, not just the first.

**How to catch:** Test with a prompt that triggers two simultaneous tool calls (e.g.
"turn on light1 and light2"). Verify both `ToolCallEvent`s are yielded and both tools
are executed.

**Rollback:** Revert `anthropic.py` and `anthropic_oauth.py` changes.

### Risk 3: History repair `_repair_tool_history()` (MEDIUM)

**Problem:** `_repair_tool_history()` scans assistant messages by calling
`_detect_function_call(content)` on the stored JSON string. After Phase 1, assistant
messages are still stored as JSON strings (format unchanged). But if the JSON format
changes, the parser breaks.

**Mitigation:** `_build_history_message()` produces the same JSON format as before
(uses `build_assistant_tool_message()` which outputs the canonical `tool_calls` array
format). `_repair_tool_history()` continues to work.

**How to catch:** Test `_repair_tool_history()` with messages generated by the new code.

### Risk 4: OpenAI `arguments` as JSON string (LOW)

**Problem:** OpenAI API returns `function.arguments` as a serialized JSON string
(not a dict). If `json.loads(raw_args)` fails, arguments become `{}` instead of the
original string.

**Mitigation:** Add logging when parse fails. Consider storing the raw string as a
fallback. In practice, OpenAI always sends valid JSON here.

**How to catch:** Unit test `_extract_response` with a tool call that has `arguments: "{\"key\": \"value\"}"`.

### Risk 5: Local model custom tool format (LOW)

**Problem:** A local model configured with `supports_tools: True` might return
tool calls in a custom format. The current code falls back to `FunctionCallParser`
for such models.

**Mitigation:** The `_legacy_string_to_response()` path is kept specifically for this.
No regression if local model returns any of the known formats.

### How to test each phase

**Phase 0:** No test needed (just type definitions).

**Phase 1 per provider:**
1. Update unit tests to expect `ProviderResponse` return type
2. Check `result.has_tool_calls` instead of `json.loads(result)["tool_calls"]`
3. Run existing integration tests

**Phase 2:**
1. `test_query_processor.py`: mock provider to return `ProviderResponse` → verify
   `FunctionCall` objects are produced correctly
2. Verify `CompletionEvent` still carries correct text
3. Run full integration test (real HA + provider mock)

**Phase 4 (cleanup):**
1. Run full test suite — no new tests needed

### Rollback plan

Each provider migration is independent. If Gemini breaks:
- Revert `gemini.py` → `supports_native_response` returns `False`
- `QueryProcessor` automatically falls back to `_legacy_string_to_response()`
- No other providers affected

If `QueryProcessor` changes break something:
- The `isinstance(raw_response, str)` guard is the safety net
- Providers returning `str` still work through legacy path

---

## Summary of Files Changed

| File | Change |
|------|--------|
| `providers/types.py` | **NEW** — `ProviderResponse`, `ProviderToolCall` |
| `providers/registry.py` | Add `supports_native_response` property; widen return type |
| `providers/base_client.py` | Widen `_extract_response` return type |
| `providers/openai.py` | `_extract_response` → returns `ProviderResponse` |
| `providers/groq.py` | **No change** (inherits from OpenAI) |
| `providers/openrouter.py` | `_extract_response` → returns `ProviderResponse` |
| `providers/gemini.py` | `_extract_response` → returns `ProviderResponse`; preserve `_gemini_raw_part` |
| `providers/gemini_oauth.py` | `_do_request` + `get_response` → return `ProviderResponse` |
| `providers/anthropic.py` | `_extract_response` → returns `ProviderResponse`; multi-tool support |
| `providers/anthropic_oauth.py` | `get_response` extraction → returns `ProviderResponse` |
| `providers/local.py` | `_extract_response` → returns `ProviderResponse` (trivial) |
| `core/query_processor.py` | `process()` + `process_stream()` non-streaming path; add helpers |
| `core/function_call_parser.py` | Docstring update (legacy scope); no logic change |
| `core/response_parser.py` | Docstring update; no logic change |
| `core/tool_call_codec.py` | **No change** |
| `tests/test_providers/*.py` | Update return type assertions |
| `tests/test_core/test_query_processor.py` | Update mock return types |

---

## Invariants This Refactor Must Preserve

1. **Gemini `thoughtSignature`**: The raw `functionCall` part must be sent back in
   history exactly as returned. Use `raw_response["_gemini_raw_part"]`.

2. **Anthropic multi-tool**: All `tool_use` blocks must be returned as separate
   `ProviderToolCall` entries and all must be executed.

3. **Anthropic pre-tool text**: `ProviderResponse.text` captures text the model
   produces before a tool call. This must be yielded as `TextEvent` before executing
   the tool.

4. **History format unchanged**: Assistant messages in conversation history continue
   to be stored as JSON strings (not `ProviderResponse` objects). The history encoding
   format is a separate concern.

5. **Streaming path untouched**: `get_response_stream()` continues to yield structured
   chunks. No changes to streaming logic.

6. **FunctionCallParser kept for local models**: Local models and custom providers
   that return OpenAI-format JSON strings still work through the legacy path.

7. **Tool name validation preserved**: After Phase 1, `QueryProcessor` still validates
   tool names against `ToolRegistry` (now done on `ProviderResponse.tool_calls` list
   instead of on parsed `FunctionCall` objects from text).
