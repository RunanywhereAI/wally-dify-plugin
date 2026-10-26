"""Live checks of the plugin's model class against https://inference.runanywhere.ai/v1.

They drive WallyLargeLanguageModel with model schemas built from this plugin's own YAML files:

    WALLY_API_KEY=sk-runa-... uv run --with "dify_plugin==0.10.2" --with pyyaml python tests/live_test.py .
"""
import base64
import json
import os
import struct
import sys
import zlib
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT))
import yaml  # noqa: E402
from dify_plugin.entities.model import AIModelEntity  # noqa: E402
from dify_plugin.entities.model.message import (  # noqa: E402
    AssistantPromptMessage,
    ImagePromptMessageContent,
    PromptMessageTool,
    TextPromptMessageContent,
    ToolPromptMessage,
    UserPromptMessage,
)
from dify_plugin.errors.model import CredentialsValidateFailedError  # noqa: E402
from models.llm.llm import WallyLargeLanguageModel  # noqa: E402

KEY = os.environ["WALLY_API_KEY"]
results = []
only = sys.argv[2:]  # optional model filter


def check(name, ok, detail=""):
    results.append((name, ok))
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


def schemas():
    out = []
    for f in sorted((ROOT / "models/llm").glob("*.yaml")):
        if f.name.startswith("_"):
            continue
        d = yaml.safe_load(f.read_text())
        d["fetch_from"] = "predefined-model"
        out.append(AIModelEntity(**d))
    return out


def llm():
    return WallyLargeLanguageModel(model_schemas=schemas())


def creds():
    return {"api_key": KEY}


WEATHER = PromptMessageTool(
    name="get_weather",
    description="Get the current weather for a city.",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)


def collect(gen):
    text, calls, usage = "", [], None
    for ch in gen:
        text += ch.delta.message.content or ""
        calls += ch.delta.message.tool_calls or []
        if ch.delta.usage:
            usage = ch.delta.usage
    return text, calls, usage


def png(r, g, b, size=64):
    raw = b"".join(b"\x00" + bytes([r, g, b]) * size for _ in range(size))

    def c(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + c(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + c(b"IDAT", zlib.compress(raw))
        + c(b"IEND", b"")
    )


def blocking(*a, **k):
    from types import SimpleNamespace
    t, calls, u = collect(llm().invoke(*a, stream=False, **k))
    return SimpleNamespace(content=t, tool_calls=calls, usage=u)


def answer(text):
    return text.split("</think>")[-1].strip()


def safe(name, fn):
    try:
        fn()
    except Exception as e:
        check(name, False, f"{type(e).__name__}: {str(e)[:200]}")


def t_creds():
    llm().validate_credentials("glm-5.3-flash", creds())
    check("validate_credentials good key", True)
    try:
        llm().validate_credentials("glm-5.3-flash", {"api_key": "sk-runa-invalid"})
        check("validate_credentials bad key rejected", False)
    except CredentialsValidateFailedError as e:
        check("validate_credentials bad key rejected", True, str(e)[:80])


def rule(m, name):
    d = yaml.safe_load((ROOT / "models/llm" / f"{m}.yaml").read_text())
    return next((r for r in d["parameter_rules"] if r["name"] == name), None)


def quiet(m):
    """Least thinking the model allows: none where thinking can be turned off."""
    return {"reasoning_effort": rule(m, "reasoning_effort")["options"][0]}


def t_model(m):
    opts = rule(m, "reasoning_effort")["options"]
    can_turn_off = opts[0] == "none"
    q = [UserPromptMessage(content="What is 17*23? Reply with just the number.")]
    r = blocking(m, creds(), q, {"max_tokens": 2048})
    check(f"{m} blocking", "391" in r.content,
          f"{r.content[:40]!r} usage={r.usage.prompt_tokens}/{r.usage.completion_tokens} ${r.usage.total_price}")
    t, _, u = collect(llm().invoke(m, creds(), q, {"max_tokens": 4096, "reasoning_effort": "high"}, stream=True))
    check(f"{m} streaming, reasoning shown as <think>", "391" in answer(t) and "<think>" in t,
          f"answer={answer(t)[:30]!r} usage={u.prompt_tokens}/{u.completion_tokens}")
    if can_turn_off:
        t, _, u = collect(llm().invoke(m, creds(), q, {"max_tokens": 512, "reasoning_effort": "none"}, stream=True))
        check(f"{m} thinking off (none)", "391" in t and "<think>" not in t, f"{t[:30]!r} out_tokens={u.completion_tokens}")
    else:
        check(f"{m} no off switch offered", "none" not in opts, str(opts))
    t, _, u = collect(llm().invoke(m, creds(), q, {"max_tokens": 4096, "reasoning_effort": opts[1] if can_turn_off else opts[0]}, stream=True))
    check(f"{m} lowest reasoning_effort", "391" in answer(t), f"out_tokens={u.completion_tokens}")
    tq = [UserPromptMessage(content="What's the weather in Paris? Use the tool.")]
    t, calls, _ = collect(llm().invoke(m, creds(), tq, {"max_tokens": 4096}, tools=[WEATHER], stream=True))
    ok = bool(calls) and calls[0].function.name == "get_weather" and "paris" in calls[0].function.arguments.lower()
    check(f"{m} streamed tool call", ok, calls[0].function.arguments if calls else t[-80:])
    if ok:
        follow = tq + [
            AssistantPromptMessage(content="", tool_calls=calls[:1]),
            ToolPromptMessage(content='{"temp_c": 21, "sky": "sunny"}', tool_call_id=calls[0].id, name="get_weather"),
        ]
        t2, _, _ = collect(llm().invoke(m, creds(), follow, {"max_tokens": 4096}, tools=[WEATHER], stream=True))
        check(f"{m} tool result round trip", "21" in answer(t2), answer(t2)[:60])
    r = blocking(m, creds(), tq, {"max_tokens": 4096}, tools=[WEATHER])
    check(f"{m} blocking tool call", bool(r.tool_calls) and r.tool_calls[0].function.name == "get_weather")
    pq = [UserPromptMessage(content="Get the weather in Paris and in Tokyo. Call the tool for both cities at once.")]
    _, calls, _ = collect(llm().invoke(m, creds(), pq, {"max_tokens": 4096, **quiet(m)}, tools=[WEATHER], stream=True))
    check(f"{m} parallel tool calls", len(calls) >= 2, f"{len(calls)} calls")
    schema = json.dumps({"name": "calc", "schema": {"type": "object", "properties": {"result": {"type": "integer"}},
                         "required": ["result"], "additionalProperties": False}, "strict": True})
    r = blocking(m, creds(), q, {"max_tokens": 4096, "response_format": "json_schema", "json_schema": schema})
    try:
        ok = json.loads(answer(r.content))["result"] == 391
    except Exception:
        ok = False
    check(f"{m} json_schema", ok, r.content[-60:])
    r = blocking(m, creds(), [UserPromptMessage(content="Return a JSON object with key a set to 1.")],
                     {"max_tokens": 4096, "response_format": "json_object", **quiet(m)})
    try:
        ok = json.loads(answer(r.content))["a"] == 1
    except Exception:
        ok = False
    check(f"{m} json_object", ok, r.content[-60:])


def vision_models():
    return [e.model for e in schemas() if any(f.value == "vision" for f in (e.features or []))]


def t_image():
    img = base64.b64encode(png(255, 0, 0)).decode()
    msg = UserPromptMessage(content=[
        TextPromptMessageContent(data="What single color fills this image? One word."),
        ImagePromptMessageContent(format="png", base64_data=img, mime_type="image/png"),
    ])
    for m in vision_models():
        t, _, _ = collect(llm().invoke(m, creds(), [msg], {"max_tokens": 2048}, stream=True))
        check(f"{m} image input", "red" in answer(t).lower(), answer(t)[:30])


def t_custom():
    cc = {"api_key": KEY, "context_size": "1048576", "max_tokens_to_sample": "131072",
          "function_calling_type": "tool_call", "vision_support": "not_support"}
    ent = llm().get_customizable_model_schema("glm-5.3-flash", dict(cc))
    check("customizable schema has reasoning effort + tool call",
          any(p.name == "reasoning_effort" for p in ent.parameter_rules)
          and any(f.value == "multi-tool-call" for f in ent.features))
    m2 = WallyLargeLanguageModel(model_schemas=[])
    t, calls, _ = collect(m2.invoke("glm-5.3-flash", dict(cc),
                                    [UserPromptMessage(content="What's the weather in Paris? Use the tool.")],
                                    {"max_tokens": 4096}, tools=[WEATHER], stream=True))
    check("customizable model streamed tool call", bool(calls), calls[0].function.arguments if calls else t[-60:])
    n = llm().get_num_tokens("glm-5.3-flash", creds(), [UserPromptMessage(content="hello world")])
    check("get_num_tokens", n > 0, str(n))


models = only or yaml.safe_load((ROOT / "models/llm/_position.yaml").read_text())
if not only:
    safe("credentials", t_creds)
for m in models:
    safe(m, lambda m=m: t_model(m))
if not only:
    safe("image", t_image)
    safe("custom", t_custom)

print(f"\n{sum(ok for _, ok in results)}/{len(results)} passed")
sys.exit(0 if all(ok for _, ok in results) else 1)
