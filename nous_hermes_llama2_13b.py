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
from transformers import AutoTokenizer

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while \
being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, \
dangerous, or illegal content. Please ensure that your responses are socially unbiased and \
positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of \
answering something not correct. If you don't know the answer to a question, please don't \
share false information."""

# max input plus output tokens need to be <1512 due to an arbitrary limitation by
MAX_OUTPUT_TOKENS = 512
MAX_INPUT_TOKENS = 1000


@dataclass
class NousHermesLlama213B(PoeBot):
    endpoint_url: str
    model_name: str
    token: str

    def __post_init__(self) -> None:
        self.client = AsyncInferenceClient(model=self.endpoint_url, token=self.token)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def get_token_count(self, input_string):
        encoded_input = self.tokenizer.encode(input_string)
        return len(encoded_input)

    def construct_prompt(self, query: QueryRequest):
        remaining_input_tokens = MAX_INPUT_TOKENS
        prompt = f"### Instruction: {DEFAULT_SYSTEM_PROMPT}\n\n"
        remaining_input_tokens -= self.get_token_count(prompt)
        response_segment = "### Response:"
        remaining_input_tokens -= self.get_token_count(response_segment)
        assert (
            remaining_input_tokens > 0
        ), "no tokens available after consuming instructions text"

        message_contents_reversed = []
        for message in reversed(query.query):
            value = None
            if message.role == "user":
                value = f"### Input: {message.content}\n\n"
            elif message.role == "bot":
                value = f"### Response: {message.content}\n\n"
            else:
                raise ValueError(f"unknown role {message.role}.")

            tokens_used_by_value = self.get_token_count(value)
            if tokens_used_by_value > remaining_input_tokens:
                break
            else:
                message_contents_reversed.append(value)
                remaining_input_tokens -= tokens_used_by_value

        prompt += "".join(reversed(message_contents_reversed))
        prompt += response_segment
        return prompt

    async def query_hf_model(self, prompt) -> Iterable[str]:
        async for token in await self.client.text_generation(
            prompt,
            stream=True,
            max_new_tokens=MAX_OUTPUT_TOKENS,
            stop_sequences=["</s>"],
        ):
            yield token

    async def get_response(self, query: QueryRequest) -> AsyncIterable[ServerSentEvent]:
        prompt = self.construct_prompt(query)
        async for token in self.query_hf_model(prompt):
            if not token.endswith("</s>"):
                yield self.text_event(token)
