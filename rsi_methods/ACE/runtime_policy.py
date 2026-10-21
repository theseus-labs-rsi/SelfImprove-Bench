"""Legacy ACE response envelopes belong to the method, not game transport."""

import json
import re
from collections.abc import Mapping

from benchmark.methods import MethodRuntimePolicy


def _parse_json_object(text):
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text)
    decoder = json.JSONDecoder()
    for candidate in candidates:
        for index, char in enumerate(candidate):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    raise ValueError("RSI agent response does not contain a JSON object")


class ACERuntimePolicy(MethodRuntimePolicy):
    def answer(self, response):
        metadata = response.metadata.get("ace") if isinstance(response.metadata, Mapping) else None
        if isinstance(metadata, Mapping) and "raw_response_text" in metadata:
            if not isinstance(response.text, str) or not response.text.strip():
                raise ValueError("ACE unwrapped response text is empty or not text")
            return response.text
        envelope = _parse_json_object(response.text)
        if "final_answer" not in envelope:
            raise ValueError("ACE response is missing final_answer")
        answer = envelope["final_answer"]
        if isinstance(answer, str):
            return answer
        if isinstance(answer, (dict, list)):
            return json.dumps(answer, ensure_ascii=False)
        raise ValueError("ACE final_answer must be a string, object, or array")

    def reasoning(self, response):
        if response.reasoning:
            return response.reasoning
        try:
            value = _parse_json_object(response.text).get("reasoning")
            return value if isinstance(value, str) else response.reasoning
        except ValueError:
            return response.reasoning
