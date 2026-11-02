# ChordDesk

**Grounded AI Customer Support Agent for Music Retail**

ChordDesk is a Python customer-support agent for the fictional **Empório da
Música**. It combines the OpenAI Responses API with deterministic application
services for catalog search, promotions, store policies, and privacy-safe order
status queries.

## Live demo

Visit [chorddesk.onrender.com](https://chorddesk.onrender.com). After a period of inactivity, the service may take around a minute or occasionally longer to wake from a Render cold start. Once the page loads, the application is ready to use. AI responses depend on OpenAI API availability and quota.

The central design principle is a strict separation of responsibilities: the
LLM handles natural-language conversation and tool orchestration, while Python
services own business facts, validation, privacy, and order-ownership
verification. The model does not receive raw CSV rows, customer records, or the
policy PDF.

## Key capabilities

- Product search by text, category, price range, and current availability
- Deterministic saleability checks from product status and stock
- Active-promotion lookup with exact decimal price calculations
- Structured parsing and lexical retrieval over the supplied policy PDF
- Order-status lookup protected by deterministic ownership verification
- Process-local multi-turn conversation history committed only after successful turns
- Generic, customer-safe verification and API failure behavior
- Explicit refusal of unrelated general-knowledge requests
- Thin FastAPI browser interface and interactive CLI over the same application core
- Independent, ephemeral browser conversations identified by opaque cookies
- Public-demo message-length and rolling-window request limits
- Responsive web UI with accessible System, Light, and Dark themes; explicit
  preferences are stored in browser `localStorage`

## Architecture

```text
Customer
  -> Web UI / FastAPI
     or CLI
  -> ConversationSession (in-memory transcript)
  -> ChordDeskAgent
  -> OpenAI Responses API
  -> function tools
  -> ToolDispatcher
  -> deterministic customer-safe application services
  -> CSV and PDF source data
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
  cleared with the CLI `/reset` command or the web UI's **Nova conversa**
  control; it is not persisted by ChordDesk.
- Browser cookies contain only a cryptographically random, opaque session
  identifier. They are `HttpOnly` and `SameSite=Lax`, with `Secure` enabled in
  production. Conversation content and customer identifiers remain in
  server-side, process-local memory.
- Public web messages are limited to 2,000 characters. Rolling-window limits
  allow 8 chat requests per browser session and 40 process-wide per 60 seconds;
  limited requests receive HTTP 429 with `Retry-After`.
- These browser controls protect a small public demo. They are not a substitute
  for production authentication, distributed rate limiting, or identity
  management.

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
- **Thin dual interfaces:** the CLI and FastAPI web application both reuse the
  same bootstrap composition, `ConversationSession`, agent, tools, and
  deterministic services.
- **Ephemeral browser sessions:** each opaque browser token maps to an independent
  in-memory session. The bounded registry expires inactive entries after one
  hour, and per-session locks prevent concurrent turns from sharing mutable
  transcript state unsafely.
- **Lightweight demo limits:** standard-library rolling windows enforce
  per-session and process-wide request limits without adding a datastore or
  external service.
- **Deployment-safe data paths:** bootstrap accepts an explicit data directory,
  then checks `CHORDDESK_DATA_DIR`, then falls back to the repository-relative
  data directory.
- **Intentional single-process hosting:** browser sessions and demo rate limits
  are process-local, so the hosted demo runs one Uvicorn process.
- **Dependency injection:** bootstrap accepts data, environment, and OpenAI
  client boundaries; the CLI and web application accept session factories.
  Tests therefore run offline with fakes and temporary fixtures.

## Project structure

```text
.
├── .python-version             # Repository runtime pin (Python 3.12.14)
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
│   ├── cli.py                    # Interactive terminal interface
│   └── web.py                    # FastAPI app, browser sessions, limits, and web UI
├── tests/                      # Offline unit and integration-style tests
├── pyproject.toml
└── uv.lock
```

## Requirements

- Python 3.12 or newer; the repository runtime is pinned to Python 3.12.14
- [uv](https://docs.astral.sh/uv/) for environment and dependency management
- An OpenAI API key for interactive use

Runtime dependencies declared in `pyproject.toml`:

- `fastapi>=0.120.0`
- `openai>=3.42.0`
- `pypdf>=6.21.4`
- `uvicorn>=0.38.0`

The development and test dependency is `httpx>=0.28.0`. The package uses
`uv_build` as its build backend and `uv.lock` for reproducible resolution.

## Running locally

Clone the public repository:

```bash
git clone https://github.com/souzacef/chorddesk.git
cd chorddesk
uv sync
export OPENAI_API_KEY="your-api-key"
```

Keep the API key in the environment. Do not commit it or place it in source
control.

### CLI

```bash
uv run chorddesk
```

Interactive commands:

- `/reset` clears the current in-memory conversation.
- `/exit` or `/quit` exits cleanly.
- Blank input is ignored; EOF and `Ctrl+C` also exit without a traceback.

### Web

```bash
uv run uvicorn chorddesk.web:app --reload
```

Open the local Uvicorn URL in a browser. Both CLI and web chat require
`OPENAI_API_KEY`; importing `chorddesk.web:app` and requesting `/health` do not
construct a conversation or make an OpenAI request.

## Deployment

The live demo runs on Render at
[chorddesk.onrender.com](https://chorddesk.onrender.com) as a single-process
Python web service.

Build command:

```bash
pip install -e .
```

Start command:

```bash
uvicorn chorddesk.web:app --host 0.0.0.0 --port $PORT
```

Render supplies `OPENAI_API_KEY` as an environment secret; it is not stored in
the repository. The service also sets:

```text
CHORDDESK_DATA_DIR=data
CHORDDESK_COOKIE_SECURE=true
```

Render checks `/health`, which does not call OpenAI. The data directory resolves
in this order: an explicit `build_session(data_dir=...)` argument, a nonblank
`CHORDDESK_DATA_DIR`, then the repository-relative `repository_data_dir()`
fallback. The single-process topology is intentional because browser sessions
and demo rate limits are process-local.

## Running tests

The test suite is fully offline: it does not require `OPENAI_API_KEY` and does
not make network requests.

```bash
uv run python -m unittest discover -s tests -v
```

At the time of this README update, the complete suite contains **171 passing
tests**.

## Representative conversations

These examples are representative of manually validated behavior and have been
edited for brevity and privacy. They are not byte-for-byte API logs.

The deployed web application was also manually validated with live OpenAI calls
for active-promotion lookup and combined promotion/policy reasoning, including
the rule that the 5% PIX discount does not stack with promotional pricing.

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

- Conversation state is process-local and is lost when the CLI or web process
  exits or restarts; browser sessions are deliberately ephemeral.
- Rate limiting is process-local rather than shared or distributed.
- There is no persistent conversation database or production customer
  authentication and identity system.
- Policy retrieval is lexical rather than embedding-based semantic retrieval.
- Source data is loaded from local CSV and PDF files at startup.
- Responses are returned after completion rather than streamed token by token.
- Product text matching is simple case-insensitive substring matching; category
  normalization is more flexible but still deterministic.
- Interactive answers depend on the configured OpenAI model and API
  availability.

Possible future work includes shared persistent session storage, distributed
rate limiting, structured observability without logging sensitive customer
messages, semantic retrieval and evaluation, production identity and
authorization, and streaming responses.

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
