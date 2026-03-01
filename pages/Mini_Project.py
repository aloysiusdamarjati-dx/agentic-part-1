"""
Mini Project: FAQ Chat Interface
Streamlit chat interface for the FAQ Agent with session state and clear history.
"""

import streamlit as st
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv

load_dotenv(override=True)

st.set_page_config(page_title="FAQ Chat - Dexa Medica", page_icon="💬")

st.title("FAQ Chat - Dexa Medica")
st.caption("Ask frequently asked questions about Dexa Medica")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Optional: Sidebar for conversations (extra points - ChatGPT-like)
if "conversations" not in st.session_state:
    st.session_state.conversations = {}
if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = "default"

# Sidebar controls
with st.sidebar:
    st.subheader("Chat Controls")

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Optional: Conversation list (ChatGPT-style) - Extra 10 points
    st.divider()
    st.subheader("Conversations")
    conv_id = st.session_state.current_conversation_id
    conv_ids = list(st.session_state.conversations.keys())
    if conv_ids:
        new_conv = st.selectbox(
            "Switch conversation",
            options=conv_ids + ["+ New conversation"],
            key="conv_selector",
        )
        if new_conv == "+ New conversation":
            import uuid
            st.session_state.current_conversation_id = str(uuid.uuid4())
            st.session_state.conversations[st.session_state.current_conversation_id] = []
            st.session_state.messages = []
            st.rerun()
        elif new_conv and new_conv != conv_id:
            st.session_state.current_conversation_id = new_conv
            st.session_state.messages = st.session_state.conversations.get(new_conv, [])[:]
            st.rerun()

        if st.button("Delete current", use_container_width=True) and conv_id in st.session_state.conversations:
            del st.session_state.conversations[conv_id]
            if st.session_state.conversations:
                st.session_state.current_conversation_id = list(st.session_state.conversations.keys())[0]
                st.session_state.messages = st.session_state.conversations[st.session_state.current_conversation_id][:]
            else:
                st.session_state.current_conversation_id = "default"
                st.session_state.messages = []
            st.rerun()
    else:
        st.write("_No conversations yet_")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
prompt = st.chat_input("Ask a question about Dexa Medica...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            import agents.FAQ as FAQ

            config = {"configurable": {"thread_id": st.session_state.current_conversation_id}}
            result = FAQ.graph.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config=config,
            )
            response = result["messages"][-1].content if result.get("messages") else "I couldn't find an answer."
            st.markdown(response)
        except Exception as e:
            st.error(f"Error: {e}")
            response = f"Sorry, an error occurred: {str(e)}"

    st.session_state.messages.append({"role": "assistant", "content": response})

    # Save to conversations
    conv_id = st.session_state.current_conversation_id
    if conv_id not in st.session_state.conversations:
        st.session_state.conversations[conv_id] = []
    st.session_state.conversations[conv_id] = st.session_state.messages

    st.rerun()
