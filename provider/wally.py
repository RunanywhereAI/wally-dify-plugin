import logging

from dify_plugin import ModelProvider
from dify_plugin.entities.model import ModelType
from dify_plugin.errors.model import CredentialsValidateFailedError

logger = logging.getLogger(__name__)


class WallyModelProvider(ModelProvider):
    def validate_provider_credentials(self, credentials: dict) -> None:
        try:
            model_instance = self.get_model_instance(ModelType.LLM)
            model_instance.validate_credentials(model="glm-5.3-flash", credentials=credentials)
        except CredentialsValidateFailedError:
            raise
        except Exception:
            logger.exception("Wally credentials validation failed")
            raise
