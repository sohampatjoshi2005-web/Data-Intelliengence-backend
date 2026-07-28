from __future__ import annotations

from abc import ABC, abstractmethod

from app.graph.state import AgentState


class BaseAgent(ABC):
    name: str

    @abstractmethod
    def run(self, state: AgentState) -> AgentState:
        raise NotImplementedError
