from __future__ import annotations

import json
import pytest
from fastapi.responses import StreamingResponse

from oas_core.manifest import LLMSample, DialogTurn, Split, LicenseInfo
from oas_core.manifest.schema import Role
from oas_server.jobs.llm_finetune import (
    _apply_asr_noise,
    _sanitize_spoken_style,
    _render_chat,
)
from oas_server.serving.cache import clear as clear_cache


# ===========================================================================
# 1. Test ASR Noise Injection
# ===========================================================================

def test_apply_asr_noise_empty() -> None:
    assert _apply_asr_noise("", 0.5) == ""
    assert _apply_asr_noise("   ", 0.5) == "   "


def test_apply_asr_noise_zero_rate() -> None:
    text = "There are two cats here."
    assert _apply_asr_noise(text, 0.0) == text


def test_apply_asr_noise_homophones() -> None:
    # Under error rate 1.0, homophones should swap
    # "there" -> "their", "here" -> "hear", "two" -> "to"
    text = "There are two cats here"
    noisy = _apply_asr_noise(text, 1.0)
    # The output should swap homophones
    # Note: "are" and "cats" will be perturbed as character swaps/omits
    words = noisy.split()
    assert words[0].lower() == "their"
    assert words[2].lower() in ("to", "too")
    assert words[4].lower() == "hear"


def test_apply_asr_noise_typos() -> None:
    # Test character swap for words longer than 2 letters
    word = "world"
    noisy = _apply_asr_noise(word, 1.0)
    assert noisy != word
    assert len(noisy) == len(word)

    # Test character omission for 1 or 2 letter words
    short = "go"
    noisy_short = _apply_asr_noise(short, 1.0)
    assert len(noisy_short) < len(short)


# ===========================================================================
# 2. Test Spoken Style Sanitizer
# ===========================================================================

def test_sanitize_spoken_style_empty() -> None:
    assert _sanitize_spoken_style("", True, True) == ""


def test_sanitize_spoken_style_markdown() -> None:
    # Test stripping code blocks and general markdown characters
    text = "Here is some code:\n```python\nprint('hello')\n```\nAnd *bold* text with `code`."
    sanitized = _sanitize_spoken_style(text, strip_markdown=True, strip_emoji=False)
    # Should strip code blocks completely and remove *, `, etc.
    assert "print('hello')" not in sanitized
    assert "bold" in sanitized
    assert "*" not in sanitized
    assert "`" not in sanitized


def test_sanitize_spoken_style_emoji() -> None:
    text = "Hello world! 🚀 😄 Have a nice day! 🌟"
    sanitized = _sanitize_spoken_style(text, strip_markdown=False, strip_emoji=True)
    assert "🚀" not in sanitized
    assert "😄" not in sanitized
    assert "🌟" not in sanitized
    # Collapse multiple spaces for simple assertion
    collapsed = " ".join(sanitized.split())
    assert collapsed == "Hello world! Have a nice day!"


# ===========================================================================
# 3. Test Chat Render Pipeline
# ===========================================================================

class MockTokenizerNoTemplate:
    def __init__(self) -> None:
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"


class MockTokenizerWithTemplate:
    def __init__(self) -> None:
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"

    def apply_chat_template(self, messages: list[dict[str, str]], tokenize: bool = False, add_generation_prompt: bool = False) -> str:
        return "<chat>" + "|".join(f"{m['role']}:{m['content']}" for m in messages) + "</chat>"


def test_render_chat_no_robustness() -> None:
    sample = LLMSample(
        id="sample-llm-1",
        license=LicenseInfo(spdx="CC0-1.0"),
        turns=[
            DialogTurn(role=Role.USER, text="Hello there!"),
            DialogTurn(role=Role.ASSISTANT, text="Hello! How can I help you today? 😄"),
        ],
        system_prompt="You are a helpful assistant.",
    )
    
    # Using a tokenizer without template (fallback test)
    tokenizer = MockTokenizerNoTemplate()
    rendered = _render_chat(sample, tokenizer)
    assert "system: You are a helpful assistant." in rendered
    assert "user: Hello there!" in rendered
    assert "assistant: Hello! How can I help you today? 😄" in rendered


def test_render_chat_with_robustness() -> None:
    sample = LLMSample(
        id="sample-llm-2",
        license=LicenseInfo(spdx="CC0-1.0"),
        turns=[
            DialogTurn(role=Role.USER, text="There is a cat here."),
            DialogTurn(role=Role.ASSISTANT, text="Here is a *bold* statement! 🚀"),
        ],
    )
    
    # 1. Under ASR noise
    robustness_asr = {
        "asr_noise_enabled": True,
        "asr_error_rate": 1.0,
        "spoken_style_enabled": False,
    }
    tokenizer = MockTokenizerWithTemplate()
    rendered = _render_chat(sample, tokenizer, robustness=robustness_asr)
    # Check that homophones or typos are introduced in user turn ("there" -> "their", "here" -> "hear")
    assert "their" in rendered.lower()
    assert "hear" in rendered.lower()
    # Check assistant turn is untouched
    assert "*bold*" in rendered
    assert "🚀" in rendered

    # 2. Under spoken style sanitization
    robustness_spoken = {
        "asr_noise_enabled": False,
        "spoken_style_enabled": True,
        "strip_markdown": True,
        "strip_emoji": True,
    }
    rendered_sp = _render_chat(sample, tokenizer, robustness=robustness_spoken)
    # Check user turn is untouched
    assert "there" in rendered_sp.lower()
    assert "here" in rendered_sp.lower()
    # Check assistant turn is sanitized
    assert "bold" in rendered_sp
    assert "*" not in rendered_sp
    assert "🚀" not in rendered_sp


# ===========================================================================
# 4. Test Chat Serving Completions (OpenAI Compatible)
# ===========================================================================

class MockLLMServer:
    def stream(self, messages: list[dict[str, str]], max_new_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.9):
        yield "Hello "
        yield "from "
        yield "mock "
        yield "LLM!"


def test_llm_chat_completions_endpoint(monkeypatch) -> None:
    # Clear serve models cache
    clear_cache()

    # Monkeypatch load_llm in serve router to return MockLLMServer
    def mock_load(version_id: str, base_model: str | None = None) -> MockLLMServer:
        return MockLLMServer()
    
    monkeypatch.setattr("oas_server.routers.serve.load_llm", mock_load)

    from oas_server.routers.serve import llm_chat, ChatCompletionIn, ChatMessage

    # A. Test non-streaming chat completions
    body_non_stream = ChatCompletionIn(
        messages=[ChatMessage(role="user", content="How are you?")],
        max_tokens=100,
        temperature=0.0,
        stream=False,
        base_model="Qwen/Qwen2.5-0.5B-Instruct"
    )
    res = llm_chat("mock-version-1", body_non_stream)
    assert res["object"] == "chat.completion"
    assert res["choices"][0]["message"]["content"] == "Hello from mock LLM!"
    assert res["choices"][0]["finish_reason"] == "stop"

    # B. Test streaming chat completions
    body_stream = ChatCompletionIn(
        messages=[ChatMessage(role="user", content="How are you?")],
        max_tokens=100,
        temperature=0.0,
        stream=True,
        base_model="Qwen/Qwen2.5-0.5B-Instruct"
    )
    res_stream = llm_chat("mock-version-2", body_stream)
    assert isinstance(res_stream, StreamingResponse)

    import asyncio

    async def consume(gen):
        out = []
        async for item in gen:
            out.append(item)
        return out

    chunks = asyncio.run(consume(res_stream.body_iterator))
    assert len(chunks) > 0

    # Parse SSE stream content
    delta_contents = []
    for line in chunks:
        for subline in line.split("\n"):
            if subline.startswith("data: ") and not subline.endswith("[DONE]"):
                data = json.loads(subline.removeprefix("data: "))
                delta = data["choices"][0]["delta"]
                if "content" in delta:
                    delta_contents.append(delta["content"])

    assert "".join(delta_contents) == "Hello from mock LLM!"
