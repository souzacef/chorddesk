# ChordDesk

**Grounded AI Customer Support Agent for Music Retail**

ChordDesk is a Python customer-support agent for the fictional **Empório da
Música**. It combines the OpenAI Responses API with deterministic application
services for catalog search, promotions, store policies, and privacy-safe order
status queries.

The central design principle is a strict separation of responsibilities: the
LLM handles natural-language conversation and tool orchestration, while Python
services own business facts, validation, privacy, and order-ownership verification. The model
does not receive raw CSV rows, customer records, or the policy PDF.

## Key capabilities

- Product search by text, category, price range, and current availability
- Deterministic saleability checks from product status and stock
- Active-promotion lookup with exact decimal price calculations
- Structured parsing and lexical retrieval over the supplied policy PDF
- Order-status lookup protected by deterministic ownership verification
- Process-local multi-turn conversation history committed only after successful turns
- Generic, customer-safe verification and API failure behavior
- Explicit refusal of unrelated general-knowledge requests
- Interactive CLI with reset and clean-exit commands

## Architecture

```text
Customer
  -> CLI
  -> ConversationSession (in-memory transcript)
  -> ChordDeskAgent
  -> OpenAI Responses API
       -> function calls
       -> ToolDispatcher
       -> customer-safe application services
       -> deterministic catalog, promotion, order, and policy layers
       -> supplied CSV and PDF files
```

The OpenAI model selects among four strict function tools. Tool calls are
validated and dispatched locally; their results come from typed Python services.
This hybrid design keeps conversational flexibility at the edge while ensuring
that prices, stock, promotion rules, policies, and order-ownership decisions do
not depend on model memory or inference.

## Grounding and safety

- Prices, stock, availability, promotions, policies, and order status are
  returned by tools backed by repository data.
- A product is saleable only when its status is `active` **and** its
  `stock_quantity` is greater than zero.
- Promotion activity and product availability are separate facts. An active
  promotion does not make an unavailable product saleable.
- An order ID alone is insufficient to release order information. The customer
  must provide exactly one verification identifier: phone or email.
- Ownership verification, including ambiguous-customer handling, is performed
  deterministically before an order-safe DTO is created.
- Every failed ownership check has the same `{"verified": false}` tool result.
  The assistant is instructed not to reveal or infer the failure cause.
- Raw customer objects are rejected by the tool-output serializer. Successful
  order results contain only the approved order and item fields, not customer
  name, ID, phone, email, or city.
- Unrelated general-knowledge requests are declined rather than answered from
  model knowledge. Store-adjacent questions remain in scope and use store data.
- Every OpenAI Responses request sets `store=False`.
- Conversation history exists only inside the current process and can be
  cleared with `/reset`; it is not persisted by ChordDesk.

## Technical decisions

- **Deterministic lexical policy retrieval:** policy sections are ranked with
  normalized tokens, heading weighting, selected Portuguese aliases, and stable
  source-order tie breaking. This is transparent and reproducible, but less
  semantically flexible than embedding-based retrieval.
- **Structured PDF parsing:** the text-based PDF is parsed into numbered,
  immutable `PolicySection` records with section titles and page ranges. Page
  headers and footers are removed before indexing.
- **OpenAI function calling:** strict, closed JSON schemas expose only product
  search, active promotions, policy search, and verified order status.
- **Customer-safe DTOs:** frozen dataclasses and immutable mappings constrain
  what crosses application boundaries and preserve exact `Decimal` values.
- **In-memory sessions:** each `ConversationSession` replays its local transcript
  to stateless Responses calls. History is committed only after a successful
  turn.
- **CLI-first interface:** the challenge is delivered as a small terminal
  application rather than an HTTP service or web UI.
- **Dependency injection:** bootstrap accepts an explicit data directory and
  OpenAI client; the CLI accepts a session factory and streams. Tests therefore
  run offline with fakes and temporary fixtures.

## Project structure

```text
.
├── data/                       # Supplied catalog, customer/order, promotion, and policy data
├── src/chorddesk/
│   ├── bootstrap.py              # Composition root and repository data paths
│   ├── catalog.py                # Catalog loading, validation, and availability
│   ├── promotions.py             # Promotion loading and exact discount calculation
│   ├── customer_orders.py        # Customer/order data and ownership verification
│   ├── policy.py                 # Structured PDF parsing
│   ├── policy_search.py          # Deterministic lexical policy retrieval
│   ├── storefront.py             # Customer-safe product and promotion services
│   ├── customer_service.py       # Policy and verified-order services
│   ├── openai_agent.py           # Tool schemas, dispatcher, orchestration, and sessions
│   └── cli.py                    # Interactive terminal interface
├── tests/                      # Offline unit and integration-style tests
├── pyproject.toml
└── uv.lock
```

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/) for environment and dependency management
- An OpenAI API key for interactive use

The runtime dependencies declared in `pyproject.toml` are the official OpenAI
Python SDK (`openai>=3.42.0`) and `pypdf>=6.21.4`. The package uses
`uv_build` as its build backend and `uv.lock` for reproducible resolution.

## Running locally

Clone the public repository:

```bash
git clone https://github.com/souzacef/chorddesk.git
cd chorddesk
uv sync
export OPENAI_API_KEY="your-api-key"
uv run chorddesk
```

Keep the API key in the environment. Do not commit it or place it in source
control.

Interactive commands:

- `/reset` clears the current in-memory conversation.
- `/exit` or `/quit` exits cleanly.
- Blank input is ignored; EOF and `Ctrl+C` also exit without a traceback.

## Running tests

The test suite is fully offline: it does not require `OPENAI_API_KEY` and does
not make network requests.

```bash
uv run python -m unittest discover -s tests -v
```

At the time of this README update, the complete suite contains **146 passing
tests**.

## Representative conversations

These examples are representative of manually validated behavior and have been
edited for brevity and privacy. They are not byte-for-byte API logs.

### Product search within a budget

> **Cliente:** Vocês têm algum violão disponível por até R$ 1.500? Me diga o preço também.
>
> **Assistente:** Sim. Encontrei várias opções disponíveis nessa faixa. Esta é uma lista abreviada:
>
> - Tagima Memphis AC-39 Nylon Natural — R$ 429,90
> - Yamaha C40 Nylon Natural — R$ 599,90
> - Yamaha F310 Aço Natural — R$ 699,90
> - Yamaha FG800 Dreadnought Natural — R$ 1.299,00
>
> Se quiser, posso filtrar por tipo de corda ou faixa de preço.

### Active promotions and PIX policy

> **Cliente:** Quais promoções estão ativas agora? Se eu pagar um produto promocional no PIX, ganho mais 5% de desconto?
>
> **Assistente:** Há quatro promoções ativas:
>
> - Kalani KAL-700T Tenor Natural: 10% — de R$ 359,00 por R$ 323,10
> - Ohana CK-20 Concert Natural: 20% — de R$ 549,00 por R$ 439,20
> - Crafter HT-100 Folk Aço Natural: 18% — de R$ 2.399,00 por R$ 1.967,18
> - Taylor 110e Dreadnought Natural: 8% — de R$ 5.999,00 por R$ 5.519,08
>
> O desconto de 5% no PIX não é cumulativo com preços promocionais.

### Secure multi-turn order lookup

The verification value below is masked and redacted for documentation.

> **Cliente:** Qual é o status do pedido 1?
>
> **Assistente:** Para consultar o pedido, informe um telefone ou um e-mail usado na compra.
>
> **Cliente:** (67) *****-1098 *(dado ocultado para documentação)*
>
> **Assistente:** O pedido 1 consta como entregue. O código de rastreamento é BRAB1234567BR.

### Online return policy

> **Cliente:** Comprei um violão online e recebi há 5 dias. Posso devolver mesmo sem defeito? Quem paga o frete da devolução?
>
> **Assistente:** Sim. Para compras online, a devolução por arrependimento pode ser solicitada em até 7 dias corridos após o recebimento, sem necessidade de justificativa. O produto deve estar na embalagem original, sem sinais de uso e com acessórios e manuais. Nesse caso, o frete de devolução é por conta da loja.

### Out-of-scope request

> **Cliente:** Qual é a capital da Austrália e quem ganhou a última Copa do Mundo?
>
> **Assistente:** Posso ajudar com produtos, pedidos, promoções e políticas da Empório da Música. Se precisar de algo relacionado à loja, é só me dizer.

## Limitations and future improvements

Current limitations:

- Conversation state is process-local and disappears when the CLI exits.
- The interface is CLI-only; there is no deployed API, web UI, or streaming.
- Policy retrieval is lexical rather than embedding-based semantic retrieval.
- Source data is loaded from local CSV and PDF files at startup.
- There is no persistent conversation store or production identity system.
- Product text matching is simple case-insensitive substring matching; category
  normalization is more flexible but still deterministic.
- Interactive answers depend on the configured OpenAI model and API
  availability.

Possible future work includes persistent sessions, an HTTP API and web UI,
structured observability without customer-message logging, richer retrieval and
evaluation, and production-grade customer identity and authorization.

## AI coding assistant disclosure

AI coding assistants were used during development for implementation support,
test generation, debugging, code review, and documentation assistance. Their
output was treated as a proposal, not as an authority: architectural and
business-rule decisions were reviewed, generated changes were inspected, and
the complete automated test suite was run repeatedly.

Live API behavior was also validated manually. Findings from that validation
led to explicit hardening of generic order-verification failure messages and
the refusal of unrelated general-knowledge requests. This feedback loop was
used to improve both privacy and scope control.

## Design assumptions

- A product is saleable only when `status == active` and
  `stock_quantity > 0`.
- Promotion activity and product availability are independent concepts.
- Order ownership requires an order ID plus exactly one verification identifier:
  phone or email.
- Phone matching is exact after trimming surrounding whitespace. Email matching
  trims whitespace and is case-insensitive.
- Any identifier resolving to more than one customer is ambiguous, and
  ambiguous resolution fails without releasing order details.
