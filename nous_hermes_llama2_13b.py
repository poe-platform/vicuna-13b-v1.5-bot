"""

Bot that lets you talk to conversational models available on HuggingFace.

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterable, Iterable

from fastapi_poe import PoeBot
from fastapi_poe.types import QueryRequest
from huggingface_hub import AsyncInferenceClient
from sse_starlette.sse import ServerSentEvent

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while \
being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, \
dangerous, or illegal content. Please ensure that your responses are socially unbiased and \
positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of \
answering something not correct. If you don't know the answer to a question, please don't \
share false information."""


@dataclass
class NousHermesLlama213B(PoeBot):
    model: str
    token: str

    def __post_init__(self) -> None:
        self.client = AsyncInferenceClient(model=self.model, token=self.token)

    def construct_prompt(self, query: QueryRequest):
        prompt = f"### Instruction: {DEFAULT_SYSTEM_PROMPT}\n\n"
        for message in query.query:
            if message.role == "user":
                prompt += f"### Input: {message.content}\n\n"
            elif message.role == "bot":
                prompt += f"### Response: {message.content}\n\n"
            else:
                raise ValueError(f"unknown role {message.role}.")
        prompt += "### Response:"
        return prompt

    async def query_hf_model(self, prompt) -> Iterable[str]:
        async for token in await self.client.text_generation(
            prompt, stream=True, max_new_tokens=1000, stop_sequences=["</s>"]
        ):
            yield token

    async def get_response(self, query: QueryRequest) -> AsyncIterable[ServerSentEvent]:
        prompt = self.construct_prompt(query)
        async for token in self.query_hf_model(prompt):
            if not token.endswith("</s>"):
                yield self.text_event(token)
