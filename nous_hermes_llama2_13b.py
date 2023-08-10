from __future__ import annotations

import json
from dataclasses import dataclass
from typing import AsyncIterable

import httpx
import httpx_sse
from fastapi_poe import PoeBot
from fastapi_poe.types import QueryRequest
from sse_starlette.sse import ServerSentEvent

BASE_URL = "https://api.together.xyz/inference"
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
    TOGETHER_API_KEY: str  # Together.ai api key

    def construct_prompt(self, query: QueryRequest):
        prompt = f"### Instruction: {DEFAULT_SYSTEM_PROMPT}\n\n"
        for message in query.query:
            if message.role == "user":
                prompt += f"### Input: {message.content}\n\n"
            elif message.role == "bot":
                prompt += f"### Response: {message.content}\n\n"
            elif message.role == "system":
                pass
            else:
                raise ValueError(f"unknown role {message.role}.")
        prompt += "### Response:"
        return prompt

    async def query_together_ai(self, prompt) -> str:
        payload = {
            "model": "NousResearch/Nous-Hermes-Llama2-13b",
            "prompt": prompt,
            "max_tokens": 1000,
            "stop": ["</s>"],
            "stream_tokens": True,
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {self.TOGETHER_API_KEY}",
        }

        async with httpx.AsyncClient() as aclient:
            async with httpx_sse.aconnect_sse(
                aclient, "POST", BASE_URL, headers=headers, json=payload
            ) as event_source:
                async for event in event_source.aiter_sse():
                    if event.data != "[DONE]":
                        token = json.loads(event.data)["choices"][0]["text"]
                        yield token

    async def get_response(self, query: QueryRequest) -> AsyncIterable[ServerSentEvent]:
        prompt = self.construct_prompt(query)
        async for word in self.query_together_ai(prompt):
            yield self.text_event(word)
