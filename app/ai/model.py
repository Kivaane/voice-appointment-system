from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.ai.nlu import NLUModelResult
from app.ai.prompts import SEMANTIC_NLU_SYSTEM_PROMPT


class AIConfigurationError(RuntimeError):
    """Raised when required AI configuration is missing."""


def get_chat_model() -> BaseChatModel:
    """Return the configured AI provider's chat model."""

    settings = get_settings()

    if settings.ai_provider == "gemini":
        if not settings.google_api_key:
            raise AIConfigurationError(
                "GOOGLE_API_KEY is missing from the .env file."
            )

        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            retries=0,
            request_timeout=10,
)

    if not settings.openai_api_key:
        raise AIConfigurationError(
            "OPENAI_API_KEY is missing from the .env file."
        )

    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
        timeout=10,
        max_retries=0,
    )


def classify_unknown_message(
    user_message: str,
) -> NLUModelResult:
    """Classify one unknown message without authorizing any action."""

    model = get_chat_model()
    structured_model = model.with_structured_output(NLUModelResult)
    result = structured_model.invoke(
        [
            SystemMessage(
                content=SEMANTIC_NLU_SYSTEM_PROMPT
            ),
            HumanMessage(content=user_message),
        ]
    )

    if isinstance(result, NLUModelResult):
        return result

    return NLUModelResult.model_validate(result)
