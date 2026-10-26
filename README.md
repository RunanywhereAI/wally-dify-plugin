# Wally

Use [Wally](https://www.runwally.com)'s fast open-weight models in Dify: chat models with tool calling for agents and workflows, reasoning, and image input on DeepSeek V4.1 Flash, DeepSeek V4.1 Flash (Fast) and Kimi K3.

Wally serves an OpenAI-compatible API at `https://inference.runanywhere.ai/v1`. This plugin points Dify at it, so you don't need to configure a generic OpenAI-compatible provider by hand.

## Setup

1. Sign in to the [RunAnywhere console](https://console.runanywhere.ai) and create a Wally API key (`sk-runa-...`).
2. In Dify, install **Wally** from the Marketplace.
3. Go to **Settings → Model Provider → Wally**, paste the API key and save. Dify checks the key with one short request.

## Usage

The models below appear in every model picker: LLM nodes, chatflows and agents.

| Model | Context | Max output | Image input |
|---|---|---|---|
| `glm-5.3-flash` (GLM-5.3 Flash) | 1M | 131,072 | no |
| `deepseek-v4.1-flash` (DeepSeek V4.1 Flash) | 1M | 65,536 | yes |
| `kimi-k3` (Kimi K3) | 1M | 65,536 | yes |
| `deepseek-v4.1-flash-fast` (DeepSeek V4.1 Flash (Fast)) | 1M | 65,536 | yes |
| `glm-5.3-fast` (GLM-5.3 (Fast)) | 1M | 131,072 | no |
| `deepseek-v4-pro-fast` (DeepSeek V4 Pro (Fast)) | 1M | 65,536 | no |
| `deepseek-v4-flash-0731` (DeepSeek V4 Flash 0731) | 1M | 65,536 | no |
| `minimax-m3` (MiniMax M3) | 512K | 131,072 | no |

- **Agents and tools:** all eight models support tool calling, including parallel and streamed tool calls.
- **Reasoning:** Dify shows the model's reasoning as a collapsible thought. Set **Reasoning effort** in the model parameters to choose how much the model thinks, or leave it empty for the model default.
  - GLM-5.3 Flash, GLM-5.3 (Fast) and MiniMax M3 always think; the lowest level is `minimal` on GLM-5.3 Flash and `low` on the other two.
  - DeepSeek V4.1 Flash, DeepSeek V4.1 Flash (Fast) and Kimi K3 think by default; `none` turns thinking off.
  - DeepSeek V4 Pro (Fast) and DeepSeek V4 Flash 0731 don't think unless you pick a level.
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
