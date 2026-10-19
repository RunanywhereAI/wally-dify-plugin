# Wally

Use [Wally](https://www.runwally.com)'s fast open-weight models in Dify: chat models with tool calling for agents and workflows, reasoning, and image input on GLM-5.3 Flash and DeepSeek V4.1 Flash.

Wally serves an OpenAI-compatible API at `https://inference.runanywhere.ai/v1`. This plugin points Dify at it, so you don't need to configure a generic OpenAI-compatible provider by hand.

## Setup

1. Sign in to the [RunAnywhere console](https://console.runanywhere.ai) and create a Wally API key (`sk-runa-...`).
2. In Dify, install **Wally** from the Marketplace.
3. Go to **Settings → Model Provider → Wally**, paste the API key and save. Dify checks the key with one short request.

## Usage

The models below appear in every model picker: LLM nodes, chatflows and agents.

| Model | Context | Max output | Image input |
|---|---|---|---|
| `glm-5.3-flash` (GLM-5.3 Flash) | 1M | 131,072 | yes |
| `deepseek-v4.1-flash` (DeepSeek V4.1 Flash) | 1M | 262,144 | yes |
| `mimo-v2.6-flash` (MiMo V2.6 Flash) | 1M | 131,072 | no |

- **Agents and tools:** all three models support tool calling, including parallel and streamed tool calls.
- **Thinking mode:** on by default. The model reasons first and Dify shows the reasoning as a collapsible thought. Turn **Thinking mode** off in the model parameters for faster, shorter replies.
- **Reasoning effort:** GLM-5.3 Flash and DeepSeek V4.1 Flash accept `low`, `high` or `max`. MiMo V2.6 Flash has thinking on or off only.
- **Structured output:** set **Response Format** to `json_object`, or to `json_schema` and paste a JSON schema.
- **Other models:** to use a Wally model that isn't listed yet, choose **Add Model**, enter its ID exactly as `GET https://inference.runanywhere.ai/v1/models` returns it, and set its context size, tool calling and image input options.

Prices are on [runwally.com/pricing](https://www.runwally.com/pricing). API reference: [docs.runanywhere.ai/wally-cloud/api](https://docs.runanywhere.ai/wally-cloud/api).

## Credentials

One Wally API key. Dify stores it encrypted and sends it as a Bearer token on each request.

## Connection requirements

Your Dify instance needs outbound HTTPS access to `inference.runanywhere.ai` (port 443). No other endpoint is called.

## Source and support

- Source repository: https://github.com/RunanywhereAI/wally-dify-plugin
- Issues: https://github.com/RunanywhereAI/wally-dify-plugin/issues
- Contact: support@runanywhere.ai
- Privacy: [PRIVACY.md](PRIVACY.md)
