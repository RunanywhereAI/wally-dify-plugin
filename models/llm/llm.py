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
                name="reasoning_effort",
                type=ParameterType.STRING,
                required=False,
                label=I18nObject(en_us="Reasoning effort"),
                help=I18nObject(
                    en_us="How much the model thinks before it answers. none turns thinking off on models that allow it."
                    " Leave empty for the model default."
                ),
                options=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
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
