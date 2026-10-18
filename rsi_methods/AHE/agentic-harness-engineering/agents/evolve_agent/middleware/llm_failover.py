# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# LLM failover middleware.
#
#  RFC-0003: LLM automatic fallback middleware
#
#  When the primary LLM provider call fails, automatically switch to a backup provider and retry.
#  Implemented as middleware for zero intrusion; the LLMCaller core code is not modified.
#
#  # Core flow
#
#  1. wrap_model_call intercepts the LLM call
#  2. When the primary provider fails, match on status_code / exception type
#  3. Create a new LLMConfig + Client and pass in modified params
#  4. Try each entry in fallback_providers order

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field

import anthropic
import openai

from nexau.archs.llm.llm_config import LLMConfig

from nexau.archs.main_sub.execution.hooks import Middleware, ModelCallFn, ModelCallParams
from nexau.archs.main_sub.execution.model_response import ModelResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FailoverTrigger:
    """Conditions that trigger failover to the next provider.

    RFC-0003: Fallback trigger condition configuration
    """

    # 1. Match HTTP status codes (e.g. 500, 502, 503, 529)
    status_codes: list[int] = field(default_factory=lambda: [500, 502, 503])
    # 2. Match exception class names (e.g. "RateLimitError", "InternalServerError")
    exception_types: list[str] = field(default_factory=lambda: list[str]())


@dataclass
class FallbackProvider:
    """A fallback LLM provider configuration.

    RFC-0003: Backup provider configuration
    """

    name: str
    # llm_config fields map directly to LLMConfig constructor arguments
    llm_config: dict[str, str | int | float | bool] = field(default_factory=lambda: dict[str, str | int | float | bool]())


@dataclass
class CircuitBreakerConfig:
    """Optional circuit breaker to avoid hammering a failing primary.

    RFC-0003: Circuit breaker configuration

    After the primary provider fails failure_threshold times in a row, for recovery_timeout_seconds
    skip the primary provider and go straight to the fallback.
    """

    failure_threshold: int = 3
    recovery_timeout_seconds: float = 60.0


# ---------------------------------------------------------------------------
# Circuit breaker (optional)
# ---------------------------------------------------------------------------


class _CircuitBreaker:
    """Simple circuit breaker for the primary provider.

    RFC-0003: Circuit breaker implementation

    State machine: CLOSED -> OPEN -> HALF_OPEN -> CLOSED
    - CLOSED: call the primary provider normally
    - OPEN: consecutive failures exceeded the threshold; skip the primary provider
    - HALF_OPEN: allow one probe after recovery_timeout
    """

    def __init__(self, config: CircuitBreakerConfig) -> None:
        self._threshold = config.failure_threshold
        self._recovery_timeout = config.recovery_timeout_seconds
        self._consecutive_failures = 0
        self._opened_at: float | None = None

    def should_skip_primary(self) -> bool:
        """Check if primary provider should be skipped."""
        if self._opened_at is None:
            return False
        elapsed = time.monotonic() - self._opened_at
        if elapsed >= self._recovery_timeout:
            # HALF_OPEN: allow one probe
            return False
        return True

    def record_success(self) -> None:
        """Record a successful call — reset breaker to CLOSED."""
        self._consecutive_failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        """Record a failed call — may trip breaker to OPEN."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._threshold:
            self._opened_at = time.monotonic()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_status_code(exc: Exception) -> int | None:
    """Extract HTTP status code from OpenAI/Anthropic SDK exceptions.

    RFC-0003: Extract the HTTP status code from an SDK exception

    Supports openai.APIStatusError.status_code and anthropic.APIStatusError.status_code.
    """
    # Both openai >= 1.x and anthropic >= 0.18 expose a status_code attribute
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code
    return None


def _build_client_for_config(config: LLMConfig) -> openai.OpenAI | anthropic.Anthropic:
    """Build an SDK client from LLMConfig.

    RFC-0003: Create the matching SDK client for the api_type
    """
    client_kwargs = config.to_client_kwargs()
    if config.api_type == "anthropic_chat_completion":
        return anthropic.Anthropic(**client_kwargs)
    return openai.OpenAI(**client_kwargs)


# ---------------------------------------------------------------------------
# Main middleware
# ---------------------------------------------------------------------------


class LLMFailoverMiddleware(Middleware):
    """LLM provider failover middleware.

    RFC-0003: LLM automatic fallback middleware

    When the primary provider fails and matches a trigger condition, automatically try the backup providers in order.
    Intercepted via wrap_model_call; the LLMCaller core code is not modified.

    YAML configuration example::

        middlewares:
          - import: nexau.archs.main_sub.execution.middleware.llm_failover:LLMFailoverMiddleware
            params:
              trigger:
                status_codes: [500, 502, 503, 529]
                exception_types: ["RateLimitError", "InternalServerError"]
              fallback_providers:
                - name: "backup-gateway"
                  llm_config:
                    base_url: "https://backup.example.com/v1"
                    api_key: "sk-backup-xxx"
                - name: "emergency"
                  llm_config:
                    model: "gpt-4o"
                    base_url: "https://emergency.example.com/v1"
                    api_key: "sk-emergency-xxx"
                    api_type: "openai_chat_completion"
              circuit_breaker:
                failure_threshold: 3
                recovery_timeout_seconds: 60
    """

    def __init__(
        self,
        *,
        fallback_providers: list[dict[str, str | dict[str, str | int | float | bool]]],
        trigger: dict[str, list[int] | list[str]] | None = None,
        circuit_breaker: dict[str, int | float] | None = None,
    ) -> None:
        # 1. Parse trigger conditions
        trigger_dict = trigger or {}
        raw_codes = trigger_dict.get("status_codes", [500, 502, 503])
        raw_exc_types = trigger_dict.get("exception_types", [])
        self._trigger = FailoverTrigger(
            status_codes=[int(c) for c in raw_codes],
            exception_types=[str(t) for t in raw_exc_types],
        )

        # 2. Parse fallback providers
        self._fallback_providers: list[FallbackProvider] = []
        for entry in fallback_providers:
            name = str(entry.get("name", f"fallback-{len(self._fallback_providers)}"))
            raw_llm = entry.get("llm_config", {})
            llm_dict = dict(raw_llm) if isinstance(raw_llm, dict) else {}
            self._fallback_providers.append(FallbackProvider(name=name, llm_config=llm_dict))

        # 3. Optional circuit breaker
        self._circuit_breaker: _CircuitBreaker | None = None
        if circuit_breaker is not None:
            cb_cfg = CircuitBreakerConfig(
                failure_threshold=int(circuit_breaker.get("failure_threshold", 3)),
                recovery_timeout_seconds=float(circuit_breaker.get("recovery_timeout_seconds", 60.0)),
            )
            self._circuit_breaker = _CircuitBreaker(cb_cfg)

    # ------------------------------------------------------------------
    # Core: wrap_model_call
    # ------------------------------------------------------------------

    def wrap_model_call(self, params: ModelCallParams, call_next: ModelCallFn) -> ModelResponse | None:
        """Intercept LLM calls and failover on matching errors.

        RFC-0003: Intercept the LLM call and fall back automatically on a matching error

        Flow:
        1. If the circuit breaker is open, skip the primary provider
        2. Try the primary provider
        3. On a failure that matches a trigger condition -> try the fallback providers in order
        4. If every provider fails -> raise the last exception
        """
        # 1. Decide whether to skip the primary provider (when the circuit breaker is open)
        skip_primary = self._circuit_breaker is not None and self._circuit_breaker.should_skip_primary()

        last_exc: Exception | None = None

        if not skip_primary:
            try:
                result = call_next(params)
                if self._circuit_breaker is not None:
                    self._circuit_breaker.record_success()
                return result
            except Exception as exc:
                last_exc = exc
                if self._circuit_breaker is not None:
                    self._circuit_breaker.record_failure()
                if not self._should_failover(exc):
                    raise
                logger.warning(
                    "LLM failover: primary provider failed (%s), trying fallback providers",
                    type(exc).__name__,
                )

        # 2. Try the fallback providers in order
        for i, provider in enumerate(self._fallback_providers):
            try:
                fallback_params = self._apply_fallback(params, provider)
                result = call_next(fallback_params)
                logger.info("LLM failover: succeeded with provider '%s'", provider.name)
                return result
            except Exception as exc:
                last_exc = exc
                is_last = i == len(self._fallback_providers) - 1
                if is_last:
                    logger.error(
                        "LLM failover: all providers exhausted, last error: %s",
                        exc,
                    )
                    raise
                logger.warning(
                    "LLM failover: provider '%s' failed (%s), trying next",
                    provider.name,
                    type(exc).__name__,
                )

        # Should not reach here, but handle defensively
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("LLM failover: Primary provider was skipped and no fallback providers were available.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _should_failover(self, exc: Exception) -> bool:
        """Check if exception matches failover trigger conditions.

        RFC-0003: Decide whether an exception matches a fallback trigger condition

        Match logic (OR):
        - status_code is in trigger.status_codes
        - the exception class name is in trigger.exception_types
        """
        # Check the status code
        status = _extract_status_code(exc)
        if status is not None and status in self._trigger.status_codes:
            return True

        # Check the exception class name
        exc_type_name = type(exc).__name__
        if exc_type_name in self._trigger.exception_types:
            return True

        return False

    def _apply_fallback(self, params: ModelCallParams, provider: FallbackProvider) -> ModelCallParams:
        """Create new ModelCallParams with fallback provider config.

        RFC-0003: Create new params that use a backup provider

        Do not modify the original params; create a copy and replace llm_config and client.
        Fields not specified in the fallback are inherited from the primary provider.
        """
        # 1. Create a copy based on the primary config
        original_config = params.llm_config
        if isinstance(original_config, LLMConfig):
            new_config = original_config.copy()
        else:
            new_config = LLMConfig()

        # 2. Override the fields specified by the fallback
        for key, value in provider.llm_config.items():
            new_config.set_param(key, value)

        # 3. Create the new client
        new_client = _build_client_for_config(new_config)

        # 4. Create the new api_params
        new_api_params = new_config.to_openai_params()
        # Keep non-config fields from the original params (such as tools, stop, etc.)
        for key in ("tools", "tool_choice", "stop"):
            if key in params.api_params and key not in new_api_params:
                new_api_params[key] = params.api_params[key]
        if params.max_tokens is not None:
            new_api_params["max_tokens"] = params.max_tokens

        # 5. Build new params (shallow-copy immutable fields, replace mutable ones)
        new_params = copy.copy(params)
        new_params.llm_config = new_config
        new_params.openai_client = new_client
        new_params.api_params = new_api_params

        return new_params
