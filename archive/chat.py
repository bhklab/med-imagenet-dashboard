"""Chat page -- Voxel, the Med-ImageNet assistant."""

from __future__ import annotations

import streamlit as st

from pages.adk_client import create_session, send_message

_MESSAGES_KEY = "chat_messages"
_SESSION_KEY = "adk_session_id"


def _ensure_session() -> str:
    """Return the current ADK session id, creating one on first use."""
    if _SESSION_KEY not in st.session_state:
        st.session_state[_SESSION_KEY] = create_session()
    return st.session_state[_SESSION_KEY]


def run() -> None:
    st.header("Voxel")
    st.caption(
        "Ask Voxel about datasets, filter collections, or run structured "
        "queries against the indexed DICOM metadata."
    )

    if _MESSAGES_KEY not in st.session_state:
        st.session_state[_MESSAGES_KEY] = []

    for msg in st.session_state[_MESSAGES_KEY]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about datasets..."):
        st.session_state[_MESSAGES_KEY].append(
            {"role": "user", "content": prompt}
        )
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                session_id = _ensure_session()
                response = send_message(session_id, prompt)
            st.markdown(response)

        st.session_state[_MESSAGES_KEY].append(
            {"role": "assistant", "content": response}
        )
        st.rerun()


run()
