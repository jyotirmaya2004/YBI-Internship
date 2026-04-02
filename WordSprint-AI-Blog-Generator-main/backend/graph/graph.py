from graph.state import BlogState
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from graph.nodes import (
    consolidate_schedule,
    get_domain_topic,
    llm_generate,
    save_markdown,
    update_articles_json,
)


def _should_skip(state: BlogState) -> str:
    if state.get("skipped"):
        return "skip"
    return "continue"


def build_graph() -> StateGraph:

    builder = StateGraph(BlogState)

    builder.add_node("consolidate_schedule", consolidate_schedule)
    builder.add_node("get_domain_topic", get_domain_topic)
    builder.add_node("llm_generate", llm_generate)
    builder.add_node("save_markdown", save_markdown)
    builder.add_node("update_articles_json", update_articles_json)

    builder.add_edge(START, "consolidate_schedule")
    builder.add_edge("consolidate_schedule", "get_domain_topic")
    builder.add_conditional_edges("get_domain_topic", _should_skip, {"skip": END, "continue": "llm_generate"})
    builder.add_edge("llm_generate", "save_markdown")
    builder.add_edge("save_markdown", "update_articles_json")
    builder.add_edge("update_articles_json", END)
    return builder.compile(checkpointer=InMemorySaver())

graph = build_graph()
