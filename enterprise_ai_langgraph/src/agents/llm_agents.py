from __future__ import annotations

from openai import OpenAI


def _chat(client: OpenAI, model_name: str, prompt: str, temperature: float = 0.2) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


def email_user_context_agent(client: OpenAI, model_name: str, email_body: str, sender_name: str) -> str:
    prompt = f"Summarize context and identify tone from {sender_name}: {email_body}"
    return _chat(client, model_name, prompt)


def intent_priority_classifier(client: OpenAI, model_name: str, email_body: str) -> str:
    prompt = (
        "Classify email type and urgency [Low/Med/High]. "
        f"Format: [TYPE] | [URGENCY]. Email: {email_body}"
    )
    return _chat(client, model_name, prompt, temperature=0.0)


def response_strategy_selector(client: OpenAI, model_name: str, classification: str, kb_context: str) -> str:
    prompt = (
        f"Given Classification: {classification} and Company Knowledge: {kb_context}, "
        "decide: Reply, Follow-up, or Escalate."
    )
    return _chat(client, model_name, prompt)


def response_ranking_agent(client: OpenAI, model_name: str, email_body: str, strategy: str) -> str:
    prompt = (
        f"Draft TWO distinct professional replies based on strategy: {strategy}. "
        f"Rank 1 and 2. Email: {email_body}"
    )
    return _chat(client, model_name, prompt)
