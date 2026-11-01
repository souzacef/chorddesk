from __future__ import annotations

import json
import unittest
from datetime import date
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock

from chorddesk.customer_orders import Customer, OrderStatus
from chorddesk.customer_service import (
    OrderItemResult,
    OrderStatusResult,
    PolicyResult,
)
from chorddesk.openai_agent import (
    DEFAULT_MODEL,
    AgentError,
    ChordDeskAgent,
    TOOLS,
    ToolDispatcher,
    ToolProtocolError,
    _json_result,
)
from chorddesk.storefront import (
    ActivePromotionResult,
    ProductResult,
    UnknownCategoryError,
)


def product_result() -> ProductResult:
    return ProductResult(
        product_id=7,
        name="Violão",
        category_name="Violões",
        price_brl=Decimal("1234.50"),
        is_available=True,
        stock_quantity=2,
        description="Cordas de nylon",
        specs={"tampo": "Spruce"},
    )


def promotion_result() -> ActivePromotionResult:
    return ActivePromotionResult(
        promotion_id=4,
        description="Oferta",
        discount_percent=Decimal("12.5"),
        product_id=7,
        product_name="Violão",
        category_name="Violões",
        original_price_brl=Decimal("1234.50"),
        discounted_price_brl=Decimal("1080.1875"),
        product_is_available=True,
        stock_quantity=2,
    )


def order_result() -> OrderStatusResult:
    return OrderStatusResult(
        order_id=101,
        status=OrderStatus.SHIPPED,
        order_date=date(2026, 10, 30),
        total_brl=Decimal("99.90"),
        payment_method="pix",
        tracking_code="TRACK-101",
        estimated_delivery=date(2026, 11, 2),
        notes=None,
        items=(OrderItemResult(product_id=7, quantity=1),),
    )


class ToolSchemaTests(unittest.TestCase):
    def test_exposes_exactly_the_four_expected_tools(self) -> None:
        self.assertEqual(
            [tool["name"] for tool in TOOLS],
            [
                "search_products",
                "get_active_promotions",
                "search_store_policies",
                "get_order_status",
            ],
        )

    def test_all_tools_are_strict_closed_schemas_with_every_property_required(
        self,
    ) -> None:
        for tool in TOOLS:
            with self.subTest(tool=tool["name"]):
                self.assertIs(tool["strict"], True)
                parameters = tool["parameters"]
                self.assertIs(parameters["additionalProperties"], False)
                self.assertEqual(
                    set(parameters["required"]), set(parameters["properties"])
                )

    def test_logically_optional_arguments_are_nullable(self) -> None:
        by_name = {tool["name"]: tool for tool in TOOLS}
        product_properties = by_name["search_products"]["parameters"]["properties"]
        for name in ("query", "category", "min_price", "max_price"):
            self.assertIn("null", product_properties[name]["type"])
        order_properties = by_name["get_order_status"]["parameters"]["properties"]
        self.assertIn("null", order_properties["phone"]["type"])
        self.assertIn("null", order_properties["email"]["type"])

    def test_policy_limit_schema_is_bounded(self) -> None:
        policy_tool = next(tool for tool in TOOLS if tool["name"] == "search_store_policies")
        limit = policy_tool["parameters"]["properties"]["limit"]
        self.assertEqual((limit["minimum"], limit["maximum"]), (1, 5))


class ToolDispatcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.storefront = Mock()
        self.policies = Mock()
        self.orders = Mock()
        self.dispatcher = ToolDispatcher(
            storefront=self.storefront,
            policies=self.policies,
            orders=self.orders,
        )

    def test_search_products_delegates_and_serializes_decimal_without_float(self) -> None:
        self.storefront.search_products.return_value = (product_result(),)
        output = self.dispatcher.dispatch(
            "search_products",
            '{"query":"violão","category":"Violões",'
            '"min_price":19.999999999999999999,"max_price":1500,'
            '"available_only":true}',
        )

        self.storefront.search_products.assert_called_once_with(
            query="violão",
            category="Violões",
            min_price=Decimal("19.999999999999999999"),
            max_price=Decimal("1500"),
            available_only=True,
        )
        payload = json.loads(output)
        self.assertEqual(payload["products"][0]["price_brl"], "1234.50")
        self.assertEqual(payload["products"][0]["specs"], {"tampo": "Spruce"})

    def test_inverted_product_price_range_is_invalid_arguments(self) -> None:
        output = self.dispatcher.dispatch(
            "search_products",
            '{"query":null,"category":null,"min_price":100,'
            '"max_price":50,"available_only":true}',
        )

        payload = json.loads(output)
        self.assertEqual(payload["error"]["code"], "invalid_arguments")
        self.assertNotEqual(payload["error"]["code"], "tool_execution_failed")
        self.storefront.search_products.assert_not_called()

    def test_get_active_promotions_delegates_and_serializes(self) -> None:
        self.storefront.get_active_promotions.return_value = (promotion_result(),)

        output = self.dispatcher.dispatch(
            "get_active_promotions", '{"available_only":false}'
        )

        self.storefront.get_active_promotions.assert_called_once_with(
            available_only=False
        )
        promotion = json.loads(output)["promotions"][0]
        self.assertEqual(promotion["discount_percent"], "12.5")
        self.assertEqual(promotion["discounted_price_brl"], "1080.1875")

    def test_search_store_policies_delegates_without_modifying_text(self) -> None:
        policy = PolicyResult("4.1", "Trocas", "Texto integral.", 3, 4, 8)
        self.policies.search_store_policies.return_value = (policy,)

        output = self.dispatcher.dispatch(
            "search_store_policies", '{"query":"troca","limit":3}'
        )

        self.policies.search_store_policies.assert_called_once_with("troca", limit=3)
        self.assertEqual(json.loads(output)["policies"][0]["content"], "Texto integral.")

    def test_blank_policy_query_is_invalid_arguments(self) -> None:
        output = self.dispatcher.dispatch(
            "search_store_policies", '{"query":"   ","limit":3}'
        )

        payload = json.loads(output)
        self.assertEqual(payload["error"]["code"], "invalid_arguments")
        self.assertNotEqual(payload["error"]["code"], "tool_execution_failed")
        self.policies.search_store_policies.assert_not_called()

    def test_successful_order_verification_serializes_without_pii(self) -> None:
        self.orders.get_order_status.return_value = order_result()

        output = self.dispatcher.dispatch(
            "get_order_status",
            '{"order_id":101,"phone":"11999999999","email":null}',
        )

        payload = json.loads(output)
        self.orders.get_order_status.assert_called_once_with(
            101, phone="11999999999", email=None
        )
        self.assertTrue(payload["verified"])
        self.assertEqual(payload["order"]["status"], "shipped")
        self.assertEqual(payload["order"]["order_date"], "2026-10-30")
        self.assertEqual(payload["order"]["total_brl"], "99.90")
        serialized = json.dumps(payload)
        for pii_name in ("customer", "customer_id", "phone", "email"):
            self.assertNotIn(pii_name, serialized)

    def test_failed_order_verification_has_only_generic_shape(self) -> None:
        self.orders.get_order_status.return_value = None

        outputs = [
            self.dispatcher.dispatch(
                "get_order_status",
                json.dumps({"order_id": order_id, "phone": phone, "email": None}),
            )
            for order_id, phone in ((999, "known"), (101, "wrong"), (101, "ambiguous"))
        ]

        self.assertEqual([json.loads(value) for value in outputs], [{"verified": False}] * 3)

    def test_both_or_neither_order_identifiers_return_safe_invocation_error(self) -> None:
        for phone, email in ((None, None), ("phone", "email@test")):
            with self.subTest(phone=phone, email=email):
                output = self.dispatcher.dispatch(
                    "get_order_status",
                    json.dumps({"order_id": 101, "phone": phone, "email": email}),
                )
                payload = json.loads(output)
                self.assertEqual(
                    payload["error"]["code"], "invalid_verification_identifiers"
                )
        self.orders.get_order_status.assert_not_called()

    def test_unknown_category_returns_safe_structured_result(self) -> None:
        self.storefront.search_products.side_effect = UnknownCategoryError("secret detail")

        output = self.dispatcher.dispatch(
            "search_products",
            '{"query":null,"category":"Bandolins","min_price":null,'
            '"max_price":null,"available_only":true}',
        )

        payload = json.loads(output)
        self.assertEqual(payload["error"]["code"], "unknown_category")
        self.assertNotIn("secret detail", output)

    def test_malformed_tool_json_fails_clearly(self) -> None:
        with self.assertRaisesRegex(ToolProtocolError, "not valid JSON"):
            self.dispatcher.dispatch("get_active_promotions", "{")

    def test_unknown_tool_name_fails_clearly(self) -> None:
        with self.assertRaisesRegex(ToolProtocolError, "unknown tool name"):
            self.dispatcher.dispatch("raw_database_query", "{}")

    def test_invalid_argument_shape_is_safe_and_does_not_call_service(self) -> None:
        output = self.dispatcher.dispatch(
            "get_active_promotions", '{"available_only":"yes"}'
        )
        self.assertEqual(json.loads(output)["error"]["code"], "invalid_arguments")
        self.storefront.get_active_promotions.assert_not_called()

    def test_unexpected_service_error_does_not_expose_details(self) -> None:
        self.storefront.get_active_promotions.side_effect = RuntimeError(
            "database password and traceback"
        )
        output = self.dispatcher.dispatch(
            "get_active_promotions", '{"available_only":true}'
        )
        self.assertEqual(json.loads(output)["error"]["code"], "tool_execution_failed")
        self.assertNotIn("password", output)
        self.assertNotIn("traceback", output)

    def test_serializer_rejects_raw_customer_objects(self) -> None:
        with self.assertRaisesRegex(TypeError, "unsupported tool output type"):
            _json_result(Customer(1, "Name", "phone", "email@test", "City"))


def function_call(call_id: str, name: str = "search_products") -> SimpleNamespace:
    return SimpleNamespace(
        type="function_call", call_id=call_id, name=name, arguments="{}"
    )


def response(*output: object, text: str = "") -> SimpleNamespace:
    return SimpleNamespace(output=list(output), output_text=text)


class FakeResponses:
    def __init__(self, queued: list[SimpleNamespace]) -> None:
        self.queued = list(queued)
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        request = dict(kwargs)
        request["input"] = list(request["input"])  # type: ignore[arg-type]
        self.requests.append(request)
        return self.queued.pop(0)


class FakeClient:
    def __init__(self, *queued: SimpleNamespace) -> None:
        self.responses = FakeResponses(list(queued))


class FakeDispatcher:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def dispatch(self, name: str, arguments: str) -> str:
        self.calls.append((name, arguments))
        return json.dumps({"called": name})


class AgentOrchestrationTests(unittest.TestCase):
    def test_zero_tool_calls_returns_output_text_immediately(self) -> None:
        client = FakeClient(response(text="Olá!"))
        dispatcher = FakeDispatcher()

        result = ChordDeskAgent(client, dispatcher).respond("Oi")

        self.assertEqual(result, "Olá!")
        self.assertEqual(len(client.responses.requests), 1)
        self.assertEqual(dispatcher.calls, [])

    def test_one_call_is_correlated_and_followed_by_second_request(self) -> None:
        call = function_call("call-1")
        client = FakeClient(response(call), response(text="Encontrei."))
        dispatcher = FakeDispatcher()

        result = ChordDeskAgent(client, dispatcher).respond("Tem violão?")

        self.assertEqual(result, "Encontrei.")
        self.assertEqual(dispatcher.calls, [("search_products", "{}")])
        second_input = client.responses.requests[1]["input"]
        self.assertIs(second_input[1], call)  # type: ignore[index]
        self.assertEqual(
            second_input[2],  # type: ignore[index]
            {
                "type": "function_call_output",
                "call_id": "call-1",
                "output": '{"called": "search_products"}',
            },
        )

    def test_all_calls_in_one_response_execute_before_next_request(self) -> None:
        first = function_call("call-1", "search_products")
        second = function_call("call-2", "get_active_promotions")
        client = FakeClient(response(first, second), response(text="Pronto."))
        dispatcher = FakeDispatcher()

        ChordDeskAgent(client, dispatcher).respond("Produtos e promoções")

        self.assertEqual(
            dispatcher.calls,
            [("search_products", "{}"), ("get_active_promotions", "{}")],
        )
        outputs = [
            item
            for item in client.responses.requests[1]["input"]  # type: ignore[union-attr]
            if isinstance(item, dict) and item.get("type") == "function_call_output"
        ]
        self.assertEqual([item["call_id"] for item in outputs], ["call-1", "call-2"])

    def test_multiple_sequential_tool_rounds_work(self) -> None:
        first = function_call("call-1", "search_products")
        second = function_call("call-2", "search_store_policies")
        client = FakeClient(
            response(first), response(second), response(text="Resposta final")
        )
        dispatcher = FakeDispatcher()

        result = ChordDeskAgent(client, dispatcher).respond("Pergunta")

        self.assertEqual(result, "Resposta final")
        self.assertEqual(len(client.responses.requests), 3)
        third_input = client.responses.requests[2]["input"]
        self.assertIn(second, third_input)  # type: ignore[operator]
        self.assertTrue(
            any(
                isinstance(item, dict) and item.get("call_id") == "call-2"
                for item in third_input  # type: ignore[union-attr]
            )
        )

    def test_all_response_output_items_are_preserved(self) -> None:
        reasoning = SimpleNamespace(type="reasoning", id="reasoning-1")
        call = function_call("call-1")
        message = SimpleNamespace(type="message", id="message-1")
        client = FakeClient(response(reasoning, call, message), response(text="Final"))

        ChordDeskAgent(client, FakeDispatcher()).respond("Pergunta")

        second_input = client.responses.requests[1]["input"]
        self.assertIs(second_input[1], reasoning)  # type: ignore[index]
        self.assertIs(second_input[2], call)  # type: ignore[index]
        self.assertIs(second_input[3], message)  # type: ignore[index]

    def test_every_request_uses_required_configuration_without_sampling_params(
        self,
    ) -> None:
        client = FakeClient(
            response(function_call("call-1")), response(text="Final")
        )

        ChordDeskAgent(client, FakeDispatcher()).respond("Pergunta")

        for request in client.responses.requests:
            self.assertEqual(request["model"], DEFAULT_MODEL)
            self.assertEqual(request["reasoning"], {"effort": "low"})
            self.assertEqual(request["tool_choice"], "auto")
            self.assertEqual(request["tools"], TOOLS)
            self.assertIs(request["store"], False)
            self.assertIn("instructions", request)
            self.assertNotIn("temperature", request)
            self.assertNotIn("top_p", request)
            self.assertNotIn("previous_response_id", request)

    def test_maximum_tool_round_guard_is_enforced(self) -> None:
        client = FakeClient(
            response(function_call("call-1")), response(function_call("call-2"))
        )
        dispatcher = FakeDispatcher()

        with self.assertRaisesRegex(AgentError, "maximum tool rounds"):
            ChordDeskAgent(client, dispatcher, max_tool_rounds=1).respond("Pergunta")

        self.assertEqual(dispatcher.calls, [("search_products", "{}")])

    def test_missing_final_output_text_fails_clearly(self) -> None:
        for text in ("", "   ", None):
            with self.subTest(text=text):
                client = FakeClient(SimpleNamespace(output=[], output_text=text))
                with self.assertRaisesRegex(AgentError, "usable output text"):
                    ChordDeskAgent(client, FakeDispatcher()).respond("Pergunta")


if __name__ == "__main__":
    unittest.main()
