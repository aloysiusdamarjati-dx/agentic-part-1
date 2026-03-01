import streamlit as st
import os
from langchain_core.messages import HumanMessage, SystemMessage
import agents.DBQNA as DBQNA
import agents.RAG as RAG
import agents.FAQ as FAQ
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.types import Command
from typing import Literal
from pydantic import BaseModel, Field

from dotenv import load_dotenv

load_dotenv(override=True)

st.title("Simple Graph with Streamlit")

DB_PATH = os.environ["DB_PATH"]

from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4.1-mini", model_provider="openai")


class BestAgent(BaseModel):
    agent_name: str = Field(
        description="The best agent to handle the request. One of: DBQNA, RAG, FAQ."
    )


class SupervisorState(MessagesState):
    user_question: str


def supervisor(state: SupervisorState) -> Command[Literal["DBQNA", "RAG", "FAQ", END]]:
    last_message = state["messages"][-1]
    instruction = [
        SystemMessage(
            content="""You receive the following question from users. Decide which agent is the most suitable for completing the task.
Delegate to DBQNA agent if users ask a question that can be answered by data inside a database.
Delegate to RAG agent if users ask a question about Dexa Medica company profile (e.g., company history, products, achievements).
Delegate to FAQ agent if users ask frequently asked questions about Dexa Medica (e.g., shipping, payment, returns, ordering process).
Respond with exactly one of: DBQNA, RAG, or FAQ."""
        )
    ]
    model_with_structure = model.with_structured_output(BestAgent)
    response = model_with_structure.invoke(instruction + [last_message])
    return Command(
        update={"user_question": last_message.content},
        goto=response.agent_name,
    )


def callRAG(state: SupervisorState) -> Command[Literal[END]]:
    prompt = state['user_question']
    response = RAG.graph.invoke({"messages":HumanMessage(content=prompt)})
    return Command(
        goto=END,
        update={"messages": response['messages'][-1]}
    )

def callDBQNA(state: SupervisorState) -> Command[Literal[END]]:
    prompt = state['user_question']
    response = DBQNA.graph.invoke({"messages":HumanMessage(content=prompt), "db_name": DB_PATH, "user_question" : prompt})
    return Command(goto=END, update={"messages": response["messages"][-1]})


def callFAQ(state: SupervisorState) -> Command[Literal[END]]:
    prompt = state["user_question"]
    config = {"configurable": {"thread_id": "lab8_faq"}}
    response = FAQ.graph.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config=config,
    )
    return Command(goto=END, update={"messages": response["messages"][-1]})


supervisor_agent = (
    StateGraph(SupervisorState)
    .add_node(supervisor)
    .add_node("RAG", callRAG)
    .add_node("DBQNA", callDBQNA)
    .add_node("FAQ", callFAQ)
    .add_edge(START, "supervisor")
    .compile(name="supervisor")
)

prompt = st.chat_input("Write your question here ... ")
if prompt:
    with st.chat_message("human"):
        st.markdown(prompt)

    final_answer = ""
    with st.chat_message("ai"):
        status_placeholder = st.empty()
        answer_placeholder = st.empty()
        status_placeholder.status(label="Process Start")
        state = "Process Start"
        for chunk, metadata in supervisor_agent.stream(
            {"messages": HumanMessage(content=prompt)}, stream_mode="messages"
        ):
            if metadata.get("langgraph_node") != state:
                node_name = metadata.get("langgraph_node", "Processing")
                status_placeholder.status(label=node_name)
                state = node_name
                final_answer = "" 
            
            if metadata.get("langgraph_node") in ("final_answer", "generate", "FAQ"):
                if hasattr(chunk, "content") and chunk.content:
                    final_answer += chunk.content
                    answer_placeholder.markdown(final_answer)
        
        status_placeholder.status(label="Complete", state='complete')

# DBQNA.graph.stream({"messages":HumanMessage(content=prompt), "db_name": DB_PATH, "user_question" : prompt}, stream_mode="messages")
# RAG.graph.stream({"messages":HumanMessage(content=prompt)}, stream_mode="messages")
            