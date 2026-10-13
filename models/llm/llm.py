from collections.abc import Generator
from typing import Optional, Union

from dify_plugin import OAICompatLargeLanguageModel
from dify_plugin.entities.model import (
    AIModelEntity,
    I18nObject,
    ModelFeature,
    ParameterRule,
    ParameterType,
)
from dify_plugin.entities.model.llm import LLMMode, LLMResult
from dify_plugin.entities.model.message import PromptMessage, PromptMessageTool

WALLY_API_BASE = "https://inference.runanywhere.ai/v1"


class WallyLargeLanguageModel(OAICompatLargeLanguageModel):
    def _invoke(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        self._add_custom_parameters(credentials)
        self._add_function_call(model, credentials)
        model_parameters = self._add_thinking_parameters(model_parameters)
        if stream:
            model_parameters["stream_options"] = {"include_usage": True}
        return super()._invoke(
            model, credentials, prompt_messages, model_parameters, tools, stop, stream, user
        )

    def validate_credentials(self, model: str, credentials: dict) -> None:
        self._add_custom_parameters(credentials)
        super().validate_credentials(model, credentials)

    def get_customizable_model_schema(self, model: str, credentials: dict) -> AIModelEntity:
        self._add_custom_parameters(credentials)
        entity = super().get_customizable_model_schema(model, credentials)
        entity.parameter_rules.append(
            ParameterRule(
                name="enable_thinking",
                type=ParameterType.BOOLEAN,
                required=False,
                default=True,
                label=I18nObject(en_us="Thinking mode"),
                help=I18nObject(
                    en_us="Let the model reason before it answers. Turn off for faster, shorter replies."
                ),
            )
        )
        return entity

    def get_num_tokens(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        tools: Optional[list[PromptMessageTool]] = None,
    ) -> int:
        self._add_custom_parameters(credentials)
        return super().get_num_tokens(model, credentials, prompt_messages, tools)

    @staticmethod
    def _add_custom_parameters(credentials: dict) -> None:
        credentials["endpoint_url"] = WALLY_API_BASE
        credentials["mode"] = LLMMode.CHAT.value

    def _add_function_call(self, model: str, credentials: dict) -> None:
        if "function_calling_type" in credentials:
            return
        model_schema = self.get_model_schema(model, credentials)
        features = (model_schema.features or []) if model_schema else []
        if ModelFeature.TOOL_CALL in features or ModelFeature.MULTI_TOOL_CALL in features:
            credentials["function_calling_type"] = "tool_call"

    @staticmethod
    def _add_thinking_parameters(model_parameters: dict) -> dict:
        model_parameters = dict(model_parameters)
        enable_thinking = model_parameters.pop("enable_thinking", None)
        if enable_thinking is None:
            return model_parameters
        model_parameters["chat_template_kwargs"] = {"enable_thinking": bool(enable_thinking)}
        if not enable_thinking:
            model_parameters.pop("reasoning_effort", None)
        return model_parameters
