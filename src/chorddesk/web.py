"""Minimal FastAPI interface for ephemeral ChordDesk conversations."""

from __future__ import annotations

import os
import secrets
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from math import ceil
from typing import Protocol

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from openai import OpenAIError
from pydantic import BaseModel

from chorddesk.bootstrap import build_session
from chorddesk.openai_agent import AgentError, ToolProtocolError


COOKIE_NAME = "chorddesk_session"
SAFE_FAILURE_MESSAGE = "Não foi possível responder agora. Tente novamente em instantes."
DEFAULT_SESSION_TTL_SECONDS = 60 * 60
DEFAULT_MAX_SESSIONS = 512
DEFAULT_MAX_MESSAGE_LENGTH = 2_000
DEFAULT_PER_SESSION_CHAT_LIMIT = 8
DEFAULT_GLOBAL_CHAT_LIMIT = 40
DEFAULT_RATE_LIMIT_WINDOW_SECONDS = 60.0
MESSAGE_TOO_LONG_DETAIL = "A mensagem deve ter no máximo 2.000 caracteres."
RATE_LIMIT_DETAIL = "Muitas mensagens em pouco tempo. Tente novamente em instantes."


class Session(Protocol):
    def respond(self, user_message: str) -> str: ...

    def reset(self) -> None: ...


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    message: str


class ResetResponse(BaseModel):
    ok: bool


class HealthResponse(BaseModel):
    status: str


@dataclass(slots=True)
class _SessionEntry:
    session: Session
    last_seen: float
    lock: threading.Lock = field(default_factory=threading.Lock)
    chat_requests: deque[float] = field(default_factory=deque)


class RollingWindowRateLimiter:
    """Atomically enforce browser and process rolling-window chat quotas."""

    def __init__(
        self,
        *,
        per_session_limit: int = DEFAULT_PER_SESSION_CHAT_LIMIT,
        global_limit: int = DEFAULT_GLOBAL_CHAT_LIMIT,
        window_seconds: float = DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if per_session_limit <= 0:
            raise ValueError("per_session_limit must be positive")
        if global_limit <= 0:
            raise ValueError("global_limit must be positive")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self._per_session_limit = per_session_limit
        self._global_limit = global_limit
        self._window_seconds = window_seconds
        self._clock = clock
        self._global_requests: deque[float] = deque()
        self._lock = threading.Lock()

    def record_or_retry_after(self, session_requests: deque[float]) -> int | None:
        """Record an allowed request or return seconds until both limits permit it."""
        with self._lock:
            now = self._clock()
            cutoff = now - self._window_seconds
            self._prune(session_requests, cutoff)
            self._prune(self._global_requests, cutoff)

            retry_times = []
            if len(session_requests) >= self._per_session_limit:
                retry_times.append(session_requests[0] + self._window_seconds - now)
            if len(self._global_requests) >= self._global_limit:
                retry_times.append(
                    self._global_requests[0] + self._window_seconds - now
                )
            if retry_times:
                return max(1, ceil(max(retry_times)))

            session_requests.append(now)
            self._global_requests.append(now)
            return None

    @staticmethod
    def _prune(requests: deque[float], cutoff: float) -> None:
        while requests and requests[0] <= cutoff:
            requests.popleft()


class SessionRegistry:
    """Thread-safe, process-local storage for short-lived browser sessions."""

    def __init__(
        self,
        session_factory: Callable[[], Session],
        *,
        ttl_seconds: float = DEFAULT_SESSION_TTL_SECONDS,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_sessions <= 0:
            raise ValueError("max_sessions must be positive")
        self._session_factory = session_factory
        self._ttl_seconds = ttl_seconds
        self._max_sessions = max_sessions
        self._clock = clock
        self._entries: dict[str, _SessionEntry] = {}
        self._lock = threading.Lock()

    def resolve(self, token: str | None) -> tuple[str, _SessionEntry, bool]:
        """Return an existing entry or create one without holding the registry lock."""
        now = self._clock()
        with self._lock:
            self._remove_expired(now)
            if token is not None and (entry := self._entries.get(token)) is not None:
                entry.last_seen = now
                return token, entry, False

        new_entry = _SessionEntry(self._session_factory(), now)
        new_token = secrets.token_urlsafe(32)
        with self._lock:
            self._remove_expired(self._clock())
            self._make_room()
            while new_token in self._entries:
                new_token = secrets.token_urlsafe(32)
            self._entries[new_token] = new_entry
        return new_token, new_entry, True

    def touch(self, token: str) -> None:
        with self._lock:
            if entry := self._entries.get(token):
                entry.last_seen = self._clock()

    def _remove_expired(self, now: float) -> None:
        expired = [
            token
            for token, entry in self._entries.items()
            if now - entry.last_seen >= self._ttl_seconds
        ]
        for token in expired:
            del self._entries[token]

    def _make_room(self) -> None:
        while len(self._entries) >= self._max_sessions:
            oldest = min(self._entries, key=lambda token: self._entries[token].last_seen)
            del self._entries[oldest]


def create_app(
    *,
    session_factory: Callable[[], Session] = build_session,
    cookie_secure: bool | None = None,
    session_ttl_seconds: float = DEFAULT_SESSION_TTL_SECONDS,
    max_sessions: int = DEFAULT_MAX_SESSIONS,
    max_message_length: int = DEFAULT_MAX_MESSAGE_LENGTH,
    per_session_chat_limit: int = DEFAULT_PER_SESSION_CHAT_LIMIT,
    global_chat_limit: int = DEFAULT_GLOBAL_CHAT_LIMIT,
    rate_limit_window_seconds: float = DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
    clock: Callable[[], float] = time.monotonic,
    environ: Mapping[str, str] = os.environ,
) -> FastAPI:
    """Create the web application with injectable offline-test boundaries."""
    if max_message_length <= 0:
        raise ValueError("max_message_length must be positive")
    if cookie_secure is None:
        cookie_secure = environ.get("CHORDDESK_COOKIE_SECURE", "").lower() in {
            "1",
            "true",
            "yes",
        }

    registry = SessionRegistry(
        session_factory,
        ttl_seconds=session_ttl_seconds,
        max_sessions=max_sessions,
        clock=clock,
    )
    rate_limiter = RollingWindowRateLimiter(
        per_session_limit=per_session_chat_limit,
        global_limit=global_chat_limit,
        window_seconds=rate_limit_window_seconds,
        clock=clock,
    )
    application = FastAPI(title="ChordDesk", docs_url=None, redoc_url=None)

    @application.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return _INDEX_HTML

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @application.post("/api/chat", response_model=ChatResponse)
    def chat(payload: ChatRequest, request: Request, response: Response) -> ChatResponse:
        message = payload.message.strip()
        if not message:
            raise HTTPException(status_code=400, detail="A mensagem não pode estar vazia.")
        if len(payload.message) > max_message_length:
            raise HTTPException(status_code=413, detail=MESSAGE_TOO_LONG_DETAIL)

        response.headers["Cache-Control"] = "no-store"
        try:
            token, entry, created = registry.resolve(request.cookies.get(COOKIE_NAME))
        except (AgentError, ToolProtocolError, OpenAIError):
            return ChatResponse(message=SAFE_FAILURE_MESSAGE)
        if created:
            _set_session_cookie(response, token, secure=cookie_secure)

        retry_after = rate_limiter.record_or_retry_after(entry.chat_requests)
        if retry_after is not None:
            raise HTTPException(
                status_code=429,
                detail=RATE_LIMIT_DETAIL,
                headers={"Retry-After": str(retry_after)},
            )

        try:
            with entry.lock:
                answer = entry.session.respond(message)
        except (AgentError, ToolProtocolError, OpenAIError):
            answer = SAFE_FAILURE_MESSAGE
        finally:
            registry.touch(token)
        return ChatResponse(message=answer)

    @application.post("/api/reset", response_model=ResetResponse)
    def reset(request: Request, response: Response) -> ResetResponse:
        response.headers["Cache-Control"] = "no-store"
        token, entry, created = registry.resolve(request.cookies.get(COOKIE_NAME))
        if created:
            _set_session_cookie(response, token, secure=cookie_secure)
        with entry.lock:
            entry.session.reset()
        registry.touch(token)
        return ResetResponse(ok=True)

    return application


def _set_session_cookie(response: Response, token: str, *, secure: bool) -> None:
    response.set_cookie(
        COOKIE_NAME,
        token,
        httponly=True,
        samesite="lax",
        secure=secure,
        path="/",
    )


app = create_app()


_INDEX_HTML = """<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ChordDesk · Empório da Música</title>
  <script>
    (() => {
      const storageKey = 'chorddesk-theme';
      const modes = ['system', 'light', 'dark'];
      const systemTheme = window.matchMedia('(prefers-color-scheme: dark)');
      let savedMode = null;
      try { savedMode = localStorage.getItem(storageKey); } catch (_) {}
      const initialMode = modes.includes(savedMode) ? savedMode : 'system';

      function applyTheme(mode) {
        const theme = mode === 'system' ? (systemTheme.matches ? 'dark' : 'light') : mode;
        document.documentElement.dataset.theme = theme;
        document.documentElement.style.colorScheme = theme;
      }

      window.chorddeskTheme = {storageKey, mode: initialMode, systemTheme, applyTheme};
      applyTheme(initialMode);
      systemTheme.addEventListener('change', () => {
        if (window.chorddeskTheme.mode === 'system') applyTheme('system');
      });
    })();
  </script>
  <style>
    :root { color-scheme: light; --ink: #22251f; --muted: #6b7067; --paper: #f5f4ef; --card: #fff; --line: #dedfd7; --accent: #315b48; --soft: #e8efe9; --control-bg: transparent; --control-text: #50564e; --suggestion-bg: #fafaf7; --input-bg: #fff; --customer-text: #fff; --focus: #9bb7a8; --error: #8c3129; --shadow: rgba(36, 42, 34, .06); }
    :root[data-theme="dark"] { color-scheme: dark; --ink: #edf1eb; --muted: #aeb8af; --paper: #141a16; --card: #1c241f; --line: #3b473f; --accent: #46785f; --soft: #29362e; --control-bg: #222b25; --control-text: #d5ddd5; --suggestion-bg: #222b25; --input-bg: #202823; --customer-text: #fff; --focus: #73aa8e; --error: #ff9d93; --shadow: rgba(0, 0, 0, .28); }
    * { box-sizing: border-box; }
    .visually-hidden { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0; }
    body { margin: 0; min-height: 100vh; background: var(--paper); color: var(--ink); font: 16px/1.5 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    main { width: min(820px, 100%); min-height: 100vh; margin: 0 auto; padding: 32px 20px; display: flex; flex-direction: column; }
    header { display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-bottom: 22px; }
    h1 { margin: 0; font-size: clamp(1.65rem, 5vw, 2.25rem); letter-spacing: -.035em; }
    .store { margin: 2px 0 0; color: var(--muted); font-size: .94rem; }
    button { font: inherit; cursor: pointer; }
    .header-actions { display: flex; align-items: center; gap: 8px; }
    .reset, .theme-select { border: 1px solid var(--line); border-radius: 9px; background: var(--control-bg); color: var(--control-text); }
    .reset { padding: 8px 12px; }
    .theme-select { min-height: 39px; padding: 7px 28px 7px 10px; font: inherit; font-size: .9rem; }
    .panel { flex: 1; min-height: 480px; border: 1px solid var(--line); border-radius: 16px; background: var(--card); box-shadow: 0 12px 32px var(--shadow); display: flex; flex-direction: column; overflow: hidden; }
    .messages { flex: 1; min-height: 300px; padding: 22px; overflow-y: auto; display: flex; flex-direction: column; gap: 14px; }
    .welcome { margin: auto; max-width: 500px; color: var(--muted); text-align: center; }
    .examples { display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; margin-top: 18px; }
    .example { border: 1px solid var(--line); border-radius: 999px; background: var(--suggestion-bg); color: var(--control-text); padding: 7px 11px; font-size: .86rem; }
    .message { max-width: 82%; padding: 10px 13px; border-radius: 13px; white-space: pre-wrap; overflow-wrap: anywhere; }
    .customer { align-self: flex-end; background: var(--accent); color: var(--customer-text); border-bottom-right-radius: 4px; }
    .assistant { align-self: flex-start; background: var(--soft); border-bottom-left-radius: 4px; }
    .composer { border-top: 1px solid var(--line); padding: 14px; display: flex; align-items: flex-end; gap: 10px; }
    textarea { flex: 1; min-height: 46px; max-height: 140px; resize: vertical; border: 1px solid var(--line); border-radius: 10px; padding: 11px 12px; color: var(--ink); background: var(--input-bg); font: inherit; }
    button:focus-visible, select:focus-visible, textarea:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }
    .send { min-height: 46px; border: 0; border-radius: 10px; background: var(--accent); color: var(--customer-text); padding: 0 18px; font-weight: 650; }
    button:disabled, select:disabled, textarea:disabled { cursor: wait; opacity: .58; }
    .error { color: var(--error); }
    @media (max-width: 600px) { main { padding: 18px 12px; } header { align-items: flex-start; } .header-actions { flex-direction: column-reverse; align-items: stretch; } .panel { min-height: calc(100vh - 112px); border-radius: 13px; } .messages { padding: 16px; } .message { max-width: 92%; } .composer { padding: 10px; } .send { padding: 0 14px; } }
  </style>
</head>
<body>
  <main>
    <header>
      <div><h1>ChordDesk</h1><p class="store">Empório da Música</p></div>
      <div class="header-actions">
        <label>
          <span class="visually-hidden">Tema de cores</span>
          <select class="theme-select" id="theme" aria-label="Tema de cores">
            <option value="system">Sistema</option>
            <option value="light">Claro</option>
            <option value="dark">Escuro</option>
          </select>
        </label>
        <button class="reset" id="reset" type="button">Nova conversa</button>
      </div>
    </header>
    <section class="panel" aria-label="Atendimento por chat">
      <div class="messages" id="messages" aria-live="polite">
        <div class="welcome" id="welcome">
          <div>Olá! Como posso ajudar com produtos, pedidos ou políticas da loja?</div>
          <div class="examples">
            <button class="example" type="button">Quais promoções estão ativas?</button>
            <button class="example" type="button">Vocês têm violões disponíveis?</button>
            <button class="example" type="button">Como funciona a troca?</button>
          </div>
        </div>
      </div>
      <form class="composer" id="composer">
        <textarea id="input" rows="1" placeholder="Digite sua mensagem…" aria-label="Mensagem" required></textarea>
        <button class="send" id="send" type="submit">Enviar</button>
      </form>
    </section>
  </main>
  <script>
    const messages = document.querySelector('#messages');
    const welcome = document.querySelector('#welcome');
    const form = document.querySelector('#composer');
    const input = document.querySelector('#input');
    const send = document.querySelector('#send');
    const reset = document.querySelector('#reset');
    const theme = document.querySelector('#theme');
    const themeState = window.chorddeskTheme;

    theme.value = themeState.mode;
    theme.addEventListener('change', () => {
      themeState.mode = theme.value;
      try { localStorage.setItem(themeState.storageKey, theme.value); } catch (_) {}
      themeState.applyTheme(theme.value);
    });

    function addMessage(text, kind, extraClass = '') {
      welcome?.remove();
      const item = document.createElement('div');
      item.className = `message ${kind} ${extraClass}`;
      item.textContent = text;
      messages.appendChild(item);
      messages.scrollTop = messages.scrollHeight;
    }

    function setBusy(busy) {
      input.disabled = busy;
      send.disabled = busy;
      reset.disabled = busy;
    }

    async function sendMessage() {
      const text = input.value.trim();
      if (!text) return;
      addMessage(text, 'customer');
      input.value = '';
      setBusy(true);
      try {
        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({message: text})
        });
        if (!response.ok) throw new Error('request failed');
        const data = await response.json();
        addMessage(data.message, 'assistant');
      } catch (_) {
        addMessage('Não consegui enviar sua mensagem. Tente novamente.', 'assistant', 'error');
      } finally {
        setBusy(false);
        input.focus();
      }
    }

    form.addEventListener('submit', event => { event.preventDefault(); sendMessage(); });
    input.addEventListener('keydown', event => {
      if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); sendMessage(); }
    });
    document.querySelectorAll('.example').forEach(button => {
      button.addEventListener('click', () => { input.value = button.textContent; input.focus(); });
    });
    reset.addEventListener('click', async () => {
      setBusy(true);
      try {
        const response = await fetch('/api/reset', {method: 'POST'});
        if (!response.ok) throw new Error('request failed');
        messages.replaceChildren();
        addMessage('Conversa reiniciada. Como posso ajudar?', 'assistant');
      } catch (_) {
        addMessage('Não consegui reiniciar a conversa. Tente novamente.', 'assistant', 'error');
      } finally {
        setBusy(false);
        input.focus();
      }
    });
  </script>
</body>
</html>
"""
