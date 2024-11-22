import logging
import time
from typing import Any, List, Literal, Optional, Union
from typing_extensions import Required, TypedDict, TypeAlias
from pydantic import BaseModel, ConfigDict, Field, model_validator
from openai.types.chat import (
    ChatCompletionContentPartParam as OpenAIChatCompletionContentPartParam,
)
from utils import random_uuid

logger = logging.getLogger(__name__)


class OpenAIBaseModel(BaseModel):
    # OpenAI API does allow extra fields
    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    @classmethod
    def __log_extra_fields__(cls, data):
        if isinstance(data, dict):
            extra_fields = data.keys() - cls.model_fields.keys()
            if extra_fields:
                logger.warning(
                    "The following fields were present in the request "
                    "but ignored: %s",
                    extra_fields,
                )
        return data


class DeltaMessage(OpenAIBaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionResponseStreamChoice(OpenAIBaseModel):
    index: int
    delta: DeltaMessage
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = None
    stop_reason: Optional[Union[int, str]] = None


class PromptTokenUsageInfo(OpenAIBaseModel):
    cached_tokens: Optional[int] = None


class UsageInfo(OpenAIBaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0
    prompt_tokens_details: Optional[PromptTokenUsageInfo] = None


class ChatCompletionStreamResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(default=None)


class CustomChatCompletionContentSimpleAudioParam(TypedDict, total=False):
    type: str
    audio_url: Required[str]


class CustomChatCompletionContentSimpleTextParam(TypedDict, total=False):
    type: str
    text: Required[str]


ChatCompletionContentPartParam: TypeAlias = Union[
    OpenAIChatCompletionContentPartParam,
    CustomChatCompletionContentSimpleTextParam,
    CustomChatCompletionContentSimpleAudioParam,
    str,
]


class ChatCompletionMessageParam(TypedDict, total=False):
    """Enables custom roles in the Chat Completion API."""

    role: Required[str]
    """The role of the message's author."""

    content: Union[str, List[ChatCompletionContentPartParam]]
    """The contents of the message."""

    name: str
    """An optional name for the participant.

    Provides the model information to differentiate between participants of the
    same role.
    """


class ChatCompletionRequest(OpenAIBaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/chat/create
    messages: List[ChatCompletionMessageParam]
    model: str
    frequency_penalty: Optional[float] = 0.0
    max_tokens: Optional[int] = Field(default=128)
    max_completion_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0

    request_id: str = Field(
        default_factory=lambda: f"{random_uuid()}",
        description=(
            "The request_id related to this request. If the caller does "
            "not set it, a random_uuid will be generated. This id is used "
            "through out the inference process and return in response."
        ),
    )
