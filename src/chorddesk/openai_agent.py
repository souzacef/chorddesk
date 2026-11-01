"""Stateless OpenAI Responses orchestration over customer-safe services."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Protocol

from chorddesk.customer_service import (
    OrderItemResult,
    OrderStatusResult,
    OrderStatusService,
    PolicyResult,
    PolicyService,
)
from chorddesk.storefront import (
    ActivePromotionResult,
    ProductResult,
    StorefrontService,
    UnknownCategoryError,
)


DEFAULT_MODEL = "gpt-6.1-terra"
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_MAX_TOOL_ROUNDS = 6

AGENT_INSTRUCTIONS = """You are the customer-service assistant for Empório da Música.
Be friendly, informal but professional. Answer in Brazilian Portuguese by default; if the
customer clearly speaks another language, use that language. Never invent store-specific
facts. Use search_products for product price, stock, availability, category, description,
or specification questions. Use get_active_promotions for current promotion questions.
Use search_store_policies for store policy or store-information questions. For a
customer-specific order status, call get_order_status only after the customer supplies
exactly one verification identifier: phone or email. If neither was supplied, ask for one,
never both. Never say verification succeeded unless the tool returns verified=true. When
get_order_status returns verified=false, give a generic verification-failure message that
asks the customer to check the information and try again. Do not say or imply that the order
exists; that the phone or email exists or does not exist; that an identifier is wrong or does
not match the order; or attribute the failure to the order ID, phone, email, ambiguity, or
ownership. If the tool results do not support an answer, say the information is unavailable
instead of guessing. Clearly unrelated general-knowledge requests are outside your scope:
do not answer them from model knowledge and do not call store tools to manufacture an
answer. Briefly and politely state that you can help with Empório da Música products,
catalog, orders, promotions, payments, shipping, returns, warranty, policies, hours, and
contact details, then invite a store-related question. Store-adjacent questions about an
unsupported item remain in scope: for example, if asked whether the store sells guitar
strings, use search_store_policies and explain the store's documented accessories policy."""


TOOLS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "name": "search_products",
        "description": (
            "Search the current Empório da Música catalog. Use whenever current "
            "store-specific product facts are needed, including price, stock, "
            "availability, category, description, specifications, or products within "
            "a budget. Null filters mean no restriction."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": ["string", "null"],
                    "description": "Product-name or description search text, or null.",
                },
                "category": {
                    "type": ["string", "null"],
                    "description": "Customer-facing catalog category name, or null.",
                },
                "min_price": {
                    "type": ["number", "null"],
                    "description": "Minimum price in BRL, or null.",
                },
                "max_price": {
                    "type": ["number", "null"],
                    "description": "Maximum price in BRL, or null.",
                },
                "available_only": {
                    "type": "boolean",
                    "description": "Whether to return only products available for sale.",
                },
            },
            "required": [
                "query",
                "category",
                "min_price",
                "max_price",
                "available_only",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_active_promotions",
        "description": (
            "Return promotions currently marked active. Product availability is a "
            "separate returned fact. This tool does not decide whether a PIX discount "
            "stacks with a promotion; use search_store_policies for stacking rules."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "available_only": {
                    "type": "boolean",
                    "description": (
                        "Whether to exclude promoted products not currently available."
                    ),
                }
            },
            "required": ["available_only"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "search_store_policies",
        "description": (
            "Search authoritative store policies and information. Use for business "
            "hours, address/contact details, payment and installment rules, returns or "
            "exchanges, shipping, PIX discounts, promotion rules, tracking, customer "
            "service, warranty, privacy, or whether the store sells an item type outside "
            "the structured catalog. Returned policy text is unmodified."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The customer's store-policy question.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": "Maximum number of matching policy sections.",
                },
            },
            "required": ["query", "limit"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_order_status",
        "description": (
            "Get customer-specific order status after verification. Provide exactly ONE "
            "of phone or email and set the other to null; never provide both. If the "
            "customer supplied neither, ask for ONE identifier before calling this tool "
            "and do not request both. A verified=false result intentionally reveals no "
            "cause. Describe it only as a generic verification failure; do not infer or "
            "name a cause involving existence, validity, matching, ambiguity, or ownership."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "integer",
                    "description": "The customer's order number.",
                },
                "phone": {
                    "type": ["string", "null"],
                    "description": "Verification phone, or null when email is used.",
                },
                "email": {
                    "type": ["string", "null"],
                    "description": "Verification email, or null when phone is used.",
                },
            },
            "required": ["order_id", "phone", "email"],
            "additionalProperties": False,
        },
        "strict": True,
    },
)


class ToolProtocolError(RuntimeError):
    """Raised when a model tool call does not follow the tool protocol."""


class AgentError(RuntimeError):
    """Raised when the local Responses orchestration cannot complete safely."""


def _product_json(result: ProductResult) -> dict[str, Any]:
    return {
        "product_id": result.product_id,
        "name": result.name,
        "category_name": result.category_name,
        "price_brl": result.price_brl,
        "is_available": result.is_available,
        "stock_quantity": result.stock_quantity,
        "description": result.description,
        "specs": result.specs,
    }


def _promotion_json(result: ActivePromotionResult) -> dict[str, Any]:
    return {
        "promotion_id": result.promotion_id,
        "description": result.description,
        "discount_percent": result.discount_percent,
        "product_id": result.product_id,
        "product_name": result.product_name,
        "category_name": result.category_name,
        "original_price_brl": result.original_price_brl,
        "discounted_price_brl": result.discounted_price_brl,
        "product_is_available": result.product_is_available,
        "stock_quantity": result.stock_quantity,
    }


def _policy_json(result: PolicyResult) -> dict[str, Any]:
    return {
        "section_number": result.section_number,
        "title": result.title,
        "content": result.content,
        "page_start": result.page_start,
        "page_end": result.page_end,
        "relevance_score": result.relevance_score,
    }


def _order_item_json(result: OrderItemResult) -> dict[str, Any]:
    return {"product_id": result.product_id, "quantity": result.quantity}


def _order_json(result: OrderStatusResult) -> dict[str, Any]:
    return {
        "order_id": result.order_id,
        "status": result.status,
        "order_date": result.order_date,
        "total_brl": result.total_brl,
        "payment_method": result.payment_method,
        "tracking_code": result.tracking_code,
        "estimated_delivery": result.estimated_delivery,
        "notes": result.notes,
        "items": result.items,
    }


def _json_safe(value: Any) -> Any:
    """Convert only approved customer-safe values to JSON-compatible values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, ProductResult):
        return _json_safe(_product_json(value))
    if isinstance(value, ActivePromotionResult):
        return _json_safe(_promotion_json(value))
    if isinstance(value, PolicyResult):
        return _json_safe(_policy_json(value))
    if isinstance(value, OrderItemResult):
        return _json_safe(_order_item_json(value))
    if isinstance(value, OrderStatusResult):
        return _json_safe(_order_json(value))
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("tool output mappings must have string keys")
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"unsupported tool output type: {type(value).__name__}")


def _json_result(value: Any) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=False, separators=(",", ":"))


def _error_result(code: str, message: str) -> str:
    return _json_result({"ok": False, "error": {"code": code, "message": message}})


@dataclass(frozen=True, slots=True)
class ToolDispatcher:
    """Dispatch model function calls exclusively to customer-safe services."""

    storefront: StorefrontService
    policies: PolicyService
    orders: OrderStatusService

    def dispatch(self, name: str, arguments: str) -> str:
        if name not in {
            "search_products",
            "get_active_promotions",
            "search_store_policies",
            "get_order_status",
        }:
            raise ToolProtocolError(f"unknown tool name: {name!r}")

        try:
            values = json.loads(arguments, parse_float=Decimal)
        except json.JSONDecodeError as exc:
            raise ToolProtocolError("tool arguments are not valid JSON") from exc
        if not isinstance(values, dict):
            raise ToolProtocolError("tool arguments must be a JSON object")

        try:
            if name == "search_products":
                return self._search_products(values)
            if name == "get_active_promotions":
                return self._get_active_promotions(values)
            if name == "search_store_policies":
                return self._search_store_policies(values)
            return self._get_order_status(values)
        except _InvalidArguments as exc:
            return _error_result("invalid_arguments", str(exc))
        except UnknownCategoryError:
            return _error_result(
                "unknown_category", "The requested category is not recognized."
            )
        except Exception:
            return _error_result(
                "tool_execution_failed", "The tool could not complete the request."
            )

    def _search_products(self, values: dict[str, Any]) -> str:
        _require_keys(
            values,
            {"query", "category", "min_price", "max_price", "available_only"},
        )
        query = _nullable_string(values["query"], "query")
        category = _nullable_string(values["category"], "category")
        min_price = _nullable_decimal(values["min_price"], "min_price")
        max_price = _nullable_decimal(values["max_price"], "max_price")
        if min_price is not None and max_price is not None and min_price > max_price:
            raise _InvalidArguments("min_price cannot be greater than max_price")
        available_only = _boolean(values["available_only"], "available_only")
        results = self.storefront.search_products(
            query=query,
            category=category,
            min_price=min_price,
            max_price=max_price,
            available_only=available_only,
        )
        return _json_result({"products": results})

    def _get_active_promotions(self, values: dict[str, Any]) -> str:
        _require_keys(values, {"available_only"})
        results = self.storefront.get_active_promotions(
            available_only=_boolean(values["available_only"], "available_only")
        )
        return _json_result({"promotions": results})

    def _search_store_policies(self, values: dict[str, Any]) -> str:
        _require_keys(values, {"query", "limit"})
        query = _string(values["query"], "query")
        if not query.strip():
            raise _InvalidArguments("query must not be blank")
        limit = _integer(values["limit"], "limit")
        if not 1 <= limit <= 5:
            raise _InvalidArguments("limit must be between 1 and 5")
        results = self.policies.search_store_policies(query, limit=limit)
        return _json_result({"policies": results})

    def _get_order_status(self, values: dict[str, Any]) -> str:
        _require_keys(values, {"order_id", "phone", "email"})
        order_id = _integer(values["order_id"], "order_id")
        phone = _nullable_string(values["phone"], "phone")
        email = _nullable_string(values["email"], "email")
        if (phone is None) == (email is None):
            return _error_result(
                "invalid_verification_identifiers",
                "Provide exactly one verification identifier: phone or email.",
            )
        result = self.orders.get_order_status(order_id, phone=phone, email=email)
        if result is None:
            return _json_result({"verified": False})
        return _json_result({"verified": True, "order": result})


class _InvalidArguments(ValueError):
    pass


def _require_keys(values: dict[str, Any], expected: set[str]) -> None:
    actual = set(values)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        parts = []
        if missing:
            parts.append(f"missing required arguments: {', '.join(missing)}")
        if unexpected:
            parts.append(f"unexpected arguments: {', '.join(unexpected)}")
        raise _InvalidArguments("; ".join(parts))


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise _InvalidArguments(f"{name} must be a string")
    return value


def _nullable_string(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _string(value, name)


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise _InvalidArguments(f"{name} must be a boolean")
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _InvalidArguments(f"{name} must be an integer")
    return value


def _nullable_decimal(value: Any, name: str) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise _InvalidArguments(f"{name} must be a number or null")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise _InvalidArguments(f"{name} must be a finite number") from exc
    if not result.is_finite():
        raise _InvalidArguments(f"{name} must be a finite number")
    return result


class _ResponsesAPI(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


class _OpenAIClient(Protocol):
    responses: _ResponsesAPI


class _Dispatcher(Protocol):
    def dispatch(self, name: str, arguments: str) -> str: ...


@dataclass(frozen=True, slots=True)
class AgentTurnResult:
    """A completed turn and the local items needed to replay it."""

    output_text: str
    replayable_items: tuple[Any, ...]


@dataclass(slots=True)
class ChordDeskAgent:
    """Run one stateless customer turn through the Responses API."""

    client: _OpenAIClient
    dispatcher: _Dispatcher
    model: str = DEFAULT_MODEL
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS
    instructions: str = AGENT_INSTRUCTIONS

    def respond(self, user_message: str) -> str:
        """Run an independent turn without retaining cross-call state."""
        return self.run_turn(user_message).output_text

    def run_turn(
        self, user_message: str, prior_items: tuple[Any, ...] = ()
    ) -> AgentTurnResult:
        """Run one turn using an explicit, locally replayable transcript."""
        if not user_message.strip():
            raise ValueError("user message must not be blank")
        if self.max_tool_rounds < 0:
            raise ValueError("max_tool_rounds cannot be negative")

        local_input = list(prior_items)
        local_input.append({"role": "user", "content": user_message})
        completed_tool_rounds = 0

        while True:
            response = self.client.responses.create(
                model=self.model,
                instructions=self.instructions,
                tools=TOOLS,
                tool_choice="auto",
                reasoning={"effort": self.reasoning_effort},
                store=False,
                input=local_input,
            )
            output = list(response.output)
            local_input.extend(output)
            calls = [
                item
                for item in output
                if getattr(item, "type", None) == "function_call"
            ]

            if not calls:
                output_text = getattr(response, "output_text", None)
                if not isinstance(output_text, str) or not output_text.strip():
                    raise AgentError("final response did not contain usable output text")
                return AgentTurnResult(output_text, tuple(local_input))

            if completed_tool_rounds >= self.max_tool_rounds:
                raise AgentError("maximum tool rounds exceeded")

            for call in calls:
                call_id = getattr(call, "call_id", None)
                name = getattr(call, "name", None)
                arguments = getattr(call, "arguments", None)
                if not all(
                    isinstance(value, str) and value for value in (call_id, name)
                ):
                    raise AgentError("function call is missing its call_id or name")
                if not isinstance(arguments, str):
                    raise AgentError("function call is missing JSON arguments")
                result = self.dispatcher.dispatch(name, arguments)
                local_input.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": result,
                    }
                )
            completed_tool_rounds += 1


@dataclass(slots=True)
class ConversationSession:
    """Own an ephemeral local transcript for explicit multi-turn continuity."""

    agent: ChordDeskAgent
    _history: tuple[Any, ...] = field(default=(), init=False, repr=False)

    @property
    def history(self) -> tuple[Any, ...]:
        """Return the committed replayable transcript as a read-only tuple."""
        return self._history

    def respond(self, user_message: str) -> str:
        result = self.agent.run_turn(user_message, self._history)
        self._history = result.replayable_items
        return result.output_text

    def reset(self) -> None:
        self._history = ()
