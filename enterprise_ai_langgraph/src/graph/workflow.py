from __future__ import annotations

from langgraph.graph import START, END, StateGraph

from ..agents.governance import governance_agent
from ..agents.llm_agents import (
    email_user_context_agent,
    intent_priority_classifier,
    response_ranking_agent,
    response_strategy_selector,
)
from ..config import settings
from ..models.state import AgentState
from ..services.tickets import create_action_ticket


class AgentRuntime:
    def __init__(self, analyzer, llm_client, chroma_collection, db_conn):
        self.analyzer = analyzer
        self.llm_client = llm_client
        self.chroma_collection = chroma_collection
        self.db_conn = db_conn


def build_workflow(runtime: AgentRuntime):
    def sanitize_node(state: AgentState) -> AgentState:
        clean = governance_agent(
            text=state["input_text"],
            user_name=state.get("user_name", "Customer"),
            user_phone=state.get("user_phone", ""),
            analyzer=runtime.analyzer,
        )
        return {"clean_text": clean}

    def classify_node(state: AgentState) -> AgentState:
        intent = intent_priority_classifier(runtime.llm_client, settings.model_name, state["clean_text"])
        priority = "High" if "High" in intent else "Medium"
        return {"intent": intent, "priority": priority}

    def ticket_node(state: AgentState) -> AgentState:
        ticket_id, due_date = create_action_ticket(
            runtime.db_conn,
            state.get("user_email", ""),
            f"Inquiry from {state.get('user_name', 'Customer')}",
            state.get("priority", "Medium"),
        )
        return {"ticket_id": ticket_id, "due_date": due_date}

    def context_node(state: AgentState) -> AgentState:
        summary = email_user_context_agent(
            runtime.llm_client,
            settings.model_name,
            state["clean_text"],
            state.get("user_name", "Customer"),
        )
        return {"context_summary": summary}

    def kb_node(state: AgentState) -> AgentState:
        kb_res = runtime.chroma_collection.query(query_texts=[state["clean_text"]], n_results=1)
        documents = kb_res.get("documents") or []
        kb_ctx = "Generic Startup Policy."
        if documents and documents[0]:
            kb_ctx = documents[0][0]
        return {"kb_context": kb_ctx}

    def strategy_node(state: AgentState) -> AgentState:
        strategy = response_strategy_selector(
            runtime.llm_client,
            settings.model_name,
            state.get("intent", "N/A"),
            state.get("kb_context", "Generic Startup Policy."),
        )
        return {"strategy": strategy}

    def draft_node(state: AgentState) -> AgentState:
        strategy_with_ticket = (
            f"{state.get('strategy', 'Reply')}. "
            f"MUST MENTION Ticket Reference: #{state.get('ticket_id', 'TBD')} in the response."
        )
        drafts = response_ranking_agent(
            runtime.llm_client,
            settings.model_name,
            state.get("clean_text", ""),
            strategy_with_ticket,
        )
        return {"drafts": drafts}

    graph = StateGraph(AgentState)
    graph.add_node("sanitize", sanitize_node)
    graph.add_node("classify", classify_node)
    graph.add_node("ticket", ticket_node)
    graph.add_node("context", context_node)
    graph.add_node("kb", kb_node)
    graph.add_node("strategy", strategy_node)
    graph.add_node("draft", draft_node)

    graph.add_edge(START, "sanitize")
    graph.add_edge("sanitize", "classify")
    graph.add_edge("classify", "ticket")
    graph.add_edge("ticket", "context")
    graph.add_edge("context", "kb")
    graph.add_edge("kb", "strategy")
    graph.add_edge("strategy", "draft")
    graph.add_edge("draft", END)

    return graph.compile()
