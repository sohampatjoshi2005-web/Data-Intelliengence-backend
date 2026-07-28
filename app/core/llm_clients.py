from __future__ import annotations

from openai import OpenAI
import boto3
import json
import os
import requests

from .config import settings
from app.services.observability import trace_generation


class LLMRouter:
    """Routes calls through Anthropic API (primary), AWS Bedrock, or Ollama (fallback)."""

    def __init__(self) -> None:
        # AWS Bedrock using environment variable credentials (not instance role)
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_REGION", settings.bedrock_region)
        
        self._bedrock = None
        if settings.bedrock_model_id:
            if aws_access_key and aws_secret_key:
                # Use explicit credentials from environment
                self._bedrock = boto3.client(
                    "bedrock-runtime",
                    region_name=aws_region,
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key
                )
            else:
                # Fallback to default credential chain (EC2 role)
                self._bedrock = boto3.client("bedrock-runtime", region_name=aws_region)
        
        # Ollama as fallback
        try:
            self._ollama = OpenAI(api_key=settings.ollama_api_key, base_url=settings.ollama_base_url)
        except Exception as e:
            print(f"DEBUG: Ollama init failed: {e}")
            self._ollama = None
        
        print(f"DEBUG: LLMRouter init. Bedrock client: {'exists' if self._bedrock else 'None'}, Model ID: {settings.bedrock_model_id}")

    def _ollama_generate(self, prompt: str) -> str:
        if not settings.ollama_base_url:
            return ""
        base_url = settings.ollama_base_url.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        resp = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": settings.ollama_model,
                "prompt": f"{prompt}\n\nReturn only the final answer.",
                "stream": False,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return str(data.get("response", "") or "").strip()

    def available_providers(self) -> list[str]:
        providers = []
        if self._bedrock and settings.bedrock_model_id:
            providers.append("bedrock")
        if self._ollama or settings.ollama_base_url:
            providers.append("ollama_local")
        return providers or ["bedrock"]

    def complete(self, prompt: str, provider: str = "bedrock", max_tokens: int = 256) -> str:
        """Complete a prompt using Bedrock or Ollama."""
        try:
            if provider == "bedrock" and self._bedrock and settings.bedrock_model_id:
                # Claude Haiku on Bedrock - optimized for speed and cost
                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": min(max_tokens, settings.bedrock_max_tokens),
                    "temperature": settings.bedrock_temperature,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                }
                resp = self._bedrock.invoke_model(
                    modelId=settings.bedrock_model_id,
                    body=json.dumps(body),
                    contentType="application/json",
                    accept="application/json",
                )
                raw = resp["body"].read().decode("utf-8")
                data = json.loads(raw)
                out = ""
                if isinstance(data, dict):
                    content = data.get("content", [])
                    if content and isinstance(content, list):
                        out = content[0].get("text", "") or ""
                trace_generation("llm_bedrock", settings.bedrock_model_id, prompt, out)
                return out or ""

            # Fallback to Ollama if Bedrock not available
            if provider == "ollama_local" and self._ollama:
                effective_max_tokens = max(max_tokens, 512)
                response = self._ollama.chat.completions.create(
                    model=settings.ollama_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=effective_max_tokens,
                )
                out = str(response.choices[0].message.content or "").strip()
                if not out:
                    out = self._ollama_generate(prompt)
                trace_generation("llm_ollama", settings.ollama_model, prompt, out)
                return out

            # If specified provider not available, try in order: bedrock -> ollama
            if self._bedrock and settings.bedrock_model_id:
                return self.complete(prompt, provider="bedrock", max_tokens=max_tokens)
            elif self._ollama:
                return self.complete(prompt, provider="ollama_local", max_tokens=max_tokens)
            else:
                return (
                    "LLM provider unavailable. Please configure AWS Bedrock credentials or ensure Ollama is running. "
                    "Falling back to rules-based behavior."
                )
        except Exception as exc:
            return (
                "LLM provider unavailable. Falling back to rules-based behavior. "
                f"Reason: {type(exc).__name__}: {exc}"
            )

    def stream_complete(self, prompt: str, provider: str = "bedrock", max_tokens: int = 256):
        """Stream a response as chunks (generator) using Bedrock or Ollama."""
        try:
            if provider == "bedrock" and self._bedrock and settings.bedrock_model_id:
                # Use Bedrock streaming API (invoke_model_with_response_stream)
                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": min(max_tokens, settings.bedrock_max_tokens),
                    "temperature": settings.bedrock_temperature,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                }
                response = self._bedrock.invoke_model_with_response_stream(
                    modelId=settings.bedrock_model_id,
                    body=json.dumps(body),
                    contentType="application/json",
                )
                
                full_response = ""
                # response["body"] is an EventStream iterator
                body_stream = response.get("body")
                if body_stream:
                    for event in body_stream:
                        try:
                            # Each event has a "chunk" with "bytes" containing the JSON
                            chunk_bytes = event.get("chunk", {}).get("bytes", b"")
                            if chunk_bytes:
                                chunk_str = chunk_bytes.decode("utf-8") if isinstance(chunk_bytes, bytes) else chunk_bytes
                                chunk = json.loads(chunk_str)
                                if chunk.get("type") == "content_block_delta":
                                    text = chunk.get("delta", {}).get("text", "")
                                    if text:
                                        full_response += text
                                        yield text
                        except (json.JSONDecodeError, AttributeError, KeyError, UnicodeDecodeError):
                            # Skip malformed events
                            pass
                
                if not full_response.strip():
                    full_response = self.complete(prompt, provider="bedrock", max_tokens=max_tokens)
                    if full_response:
                        yield full_response

                trace_generation("llm_bedrock_stream", settings.bedrock_model_id, prompt, full_response)
                return

            # Fallback to Ollama streaming
            if provider == "ollama_local" and self._ollama:
                stream = self._ollama.chat.completions.create(
                    model=settings.ollama_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=max_tokens,
                    stream=True,
                )
                
                full_response = ""
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        yield content
                
                if not full_response.strip():
                    full_response = self.complete(prompt, provider="ollama_local", max_tokens=max_tokens)
                    if full_response:
                        yield full_response

                trace_generation("llm_ollama_stream", settings.ollama_model, prompt, full_response)
                return

            # Fallback: non-streaming response
            result = self.complete(prompt, provider, max_tokens)
            yield result
        except Exception as exc:
            error_msg = f"LLM provider unavailable: {type(exc).__name__}: {exc}"
            yield error_msg
