from __future__ import annotations

from app.agents.analytics_code_generation import AnalyticsCodeGenerationAgent
from app.agents.analytics_execution import AnalyticsExecutionAgent
from app.agents.analytics_insights import AnalyticsInsightsAgent
from app.agents.analytics_query_understanding import AnalyticsQueryUnderstandingAgent
from app.agents.analytics_reasoning import AnalyticsReasoningAgent
from app.agents.data_understanding import DataUnderstandingAgent
from app.agents.deployment import DeploymentAgent
from app.agents.evaluator import EvaluatorAgent
from app.agents.model_training import ModelTrainingAgent
from app.agents.pipeline_generator import PipelineGeneratorAgent
from app.agents.problem_classification import ProblemClassificationAgent
from app.graph.state import AgentState

try:
    from langgraph.graph import END, StateGraph
except Exception:
    END = "END"
    StateGraph = None


def build_workflow():
    if StateGraph is None:
        return None

    graph = StateGraph(AgentState)

    data_agent = DataUnderstandingAgent()
    problem_agent = ProblemClassificationAgent()
    pipeline_agent = PipelineGeneratorAgent()
    train_agent = ModelTrainingAgent()
    eval_agent = EvaluatorAgent()
    deploy_agent = DeploymentAgent()

    graph.add_node(data_agent.name, data_agent.run)
    graph.add_node(problem_agent.name, problem_agent.run)
    graph.add_node(pipeline_agent.name, pipeline_agent.run)
    graph.add_node(train_agent.name, train_agent.run)
    graph.add_node(eval_agent.name, eval_agent.run)
    graph.add_node(deploy_agent.name, deploy_agent.run)

    graph.set_entry_point(data_agent.name)
    graph.add_edge(data_agent.name, problem_agent.name)
    graph.add_edge(problem_agent.name, pipeline_agent.name)
    graph.add_edge(pipeline_agent.name, train_agent.name)
    graph.add_edge(train_agent.name, eval_agent.name)
    graph.add_edge(eval_agent.name, deploy_agent.name)
    graph.add_edge(deploy_agent.name, END)

    return graph.compile()


def build_analytics_workflow():
    if StateGraph is None:
        return None

    graph = StateGraph(AgentState)
    q = AnalyticsQueryUnderstandingAgent()
    cg = AnalyticsCodeGenerationAgent()
    ex = AnalyticsExecutionAgent()
    rs = AnalyticsReasoningAgent()

    graph.add_node(q.name, q.run)
    graph.add_node(cg.name, cg.run)
    graph.add_node(ex.name, ex.run)
    graph.add_node(rs.name, rs.run)

    graph.set_entry_point(q.name)
    graph.add_edge(q.name, cg.name)
    graph.add_edge(cg.name, ex.name)
    graph.add_edge(ex.name, rs.name)
    graph.add_edge(rs.name, END)
    return graph.compile()


def run_sequential_fallback(state: AgentState) -> AgentState:
    for agent in [
        DataUnderstandingAgent(),
        ProblemClassificationAgent(),
        PipelineGeneratorAgent(),
        ModelTrainingAgent(),
        EvaluatorAgent(),
        DeploymentAgent(),
    ]:
        state = agent.run(state)
    return state


def run_analytics_sequential_fallback(state: AgentState) -> AgentState:
    for agent in [
        AnalyticsQueryUnderstandingAgent(),
        AnalyticsCodeGenerationAgent(),
        AnalyticsExecutionAgent(),
        AnalyticsReasoningAgent(),
    ]:
        state = agent.run(state)
    return state


def run_analytics_insights(state: AgentState) -> AgentState:
    return AnalyticsInsightsAgent().run(state)
