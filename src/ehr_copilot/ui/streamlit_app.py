"""Streamlit demo chat interface for the EHR Copilot.

Run with:
    streamlit run src/ehr_copilot/ui/streamlit_app.py

Expects the FastAPI backend to be available at http://localhost:8000.
"""

from __future__ import annotations

import json
import re
import time

import httpx
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE = "http://localhost:8000"
REQUEST_TIMEOUT = 180.0  # seconds -- pipeline can be slow on first call

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="EHR Copilot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal clinical-looking custom CSS.
st.markdown(
    """
    <style>
    /* Tone down default Streamlit padding */
    .block-container { padding-top: 2rem; }

    /* Chat message styling */
    .citation-marker {
        background-color: #e3f2fd;
        border-radius: 4px;
        padding: 1px 5px;
        font-size: 0.85em;
        font-weight: 600;
        color: #1565c0;
    }

    /* Verdict badges */
    .verdict-approved { color: #2e7d32; font-weight: 700; }
    .verdict-revised  { color: #ef6c00; font-weight: 700; }
    .verdict-abstained { color: #c62828; font-weight: 700; }

    /* Sidebar section headers */
    .sidebar-header { font-size: 1.1em; font-weight: 600; margin-top: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_state():
    """Initialise Streamlit session state keys."""
    defaults = {
        "patient_id": None,
        "patient_name": None,
        "session_id": None,
        "chunk_count": 0,
        "resource_counts": {},
        "messages": [],  # list of {"role": "user"|"assistant", "content": ..., "metadata": ...}
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _api_get(path: str, timeout: float = 10.0) -> dict | list | None:
    """Perform a GET request against the backend."""
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(f"{API_BASE}{path}")
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as exc:
        st.error(f"API error: {exc}")
        return None


def _api_post(path: str, payload: dict, timeout: float = REQUEST_TIMEOUT) -> dict | None:
    """Perform a POST request against the backend."""
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(f"{API_BASE}{path}", json=payload)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            pass
        st.error(f"API error ({exc.response.status_code}): {detail or exc}")
        return None
    except httpx.HTTPError as exc:
        st.error(f"Connection error: {exc}")
        return None


def _api_delete(path: str, timeout: float = 10.0) -> dict | None:
    """Perform a DELETE request against the backend."""
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.delete(f"{API_BASE}{path}")
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as exc:
        st.error(f"API error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Sidebar -- patient management
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("EHR Copilot")
    st.markdown("---")

    # Health check indicator
    health = _api_get("/health")
    if health:
        status_emoji = "🟢" if health.get("status") == "ok" else "🔴"
        llm_emoji = "🟢" if health.get("llm_available") else "🟡"
        st.caption(f"{status_emoji} Backend: {health.get('status', 'unknown')}  |  {llm_emoji} LLM")
        st.caption(f"Version: {health.get('version', '?')}")
    else:
        st.warning("Backend not reachable at " + API_BASE)

    st.markdown("---")
    st.markdown('<div class="sidebar-header">Load Patient</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload FHIR Bundle JSON",
        type=["json"],
        help="Select a Synthea-generated FHIR R4 Bundle JSON file.",
    )

    file_path_input = st.text_input(
        "Or enter file path on server",
        placeholder="/path/to/fhir_bundle.json",
    )

    if st.button("Load Patient", type="primary", use_container_width=True):
        target_path = None

        if uploaded_file is not None:
            # Save uploaded file to a temp location so the backend can read it.
            import tempfile, os

            tmp_dir = tempfile.mkdtemp(prefix="ehr_copilot_")
            tmp_path = os.path.join(tmp_dir, uploaded_file.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            target_path = tmp_path
        elif file_path_input.strip():
            target_path = file_path_input.strip()
        else:
            st.warning("Please upload a file or enter a file path.")

        if target_path:
            with st.spinner("Parsing and indexing..."):
                result = _api_post("/patient/load", {"file_path": target_path})
            if result:
                st.session_state.patient_id = result["patient_id"]
                st.session_state.patient_name = result["display_name"]
                st.session_state.session_id = result["session_id"]
                st.session_state.chunk_count = result["chunk_count"]
                st.session_state.resource_counts = result["resource_counts"]
                st.session_state.messages = []
                st.success(f"Loaded: {result['display_name']}")
                st.rerun()

    # Currently loaded patient summary
    if st.session_state.patient_id:
        st.markdown("---")
        st.markdown('<div class="sidebar-header">Current Patient</div>', unsafe_allow_html=True)
        st.markdown(f"**{st.session_state.patient_name}**")
        st.caption(f"ID: `{st.session_state.patient_id}`")
        st.caption(f"Session: `{st.session_state.session_id}`")
        st.caption(f"Indexed chunks: {st.session_state.chunk_count}")

        if st.session_state.resource_counts:
            with st.expander("Resource Counts"):
                for rtype, count in st.session_state.resource_counts.items():
                    st.text(f"  {rtype}: {count}")

        if st.button("Unload Patient", use_container_width=True):
            _api_delete(f"/patient/{st.session_state.patient_id}")
            st.session_state.patient_id = None
            st.session_state.patient_name = None
            st.session_state.session_id = None
            st.session_state.chunk_count = 0
            st.session_state.resource_counts = {}
            st.session_state.messages = []
            st.rerun()

    # Loaded patients list
    st.markdown("---")
    st.markdown('<div class="sidebar-header">All Loaded Patients</div>', unsafe_allow_html=True)
    patients_list = _api_get("/patient/list")
    if patients_list:
        for p in patients_list:
            st.caption(f"- {p.get('display_name', p.get('patient_id'))} ({p.get('chunk_count', 0)} chunks)")
    else:
        st.caption("No patients loaded")


# ---------------------------------------------------------------------------
# Main area -- chat interface
# ---------------------------------------------------------------------------

st.header("Clinical Query Chat")

if not st.session_state.patient_id:
    st.info("Load a patient from the sidebar to start asking clinical questions.")
    st.stop()

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            # Assistant response with structured output
            _render_answer(msg) if "metadata" in msg else st.markdown(msg["content"])


def _render_answer(msg: dict):
    """Render an assistant message with citations and expandable details."""
    meta = msg.get("metadata", {})
    answer_text = msg["content"]

    # Verdict badge
    verdict = meta.get("verdict", "")
    verdict_class = f"verdict-{verdict}" if verdict else ""
    if verdict:
        st.markdown(
            f'<span class="{verdict_class}">{verdict.upper()}</span> '
            f'(confidence: {meta.get("confidence", 0):.0%}, '
            f'latency: {meta.get("latency_ms", 0):.0f} ms)',
            unsafe_allow_html=True,
        )

    # Abstention notice
    if verdict == "abstained" and meta.get("abstention_reason"):
        st.warning(f"Abstention: {meta['abstention_reason']}")

    # Answer text (with citation markers highlighted)
    display_text = _highlight_citations(answer_text)
    st.markdown(display_text, unsafe_allow_html=True)

    # Expandable sections
    evidence_pack = meta.get("evidence_pack")

    if evidence_pack:
        # Citations / references
        refs = evidence_pack.get("formatted_references", "")
        if refs:
            with st.expander("References"):
                st.text(refs)

        # Evidence sources
        sources = evidence_pack.get("source_chunks", {})
        if sources:
            with st.expander(f"Evidence Sources ({len(sources)} chunks)"):
                for cid, chunk_data in sources.items():
                    chunk_text = chunk_data.get("text", "")[:300]
                    source_label = chunk_data.get("metadata", {}).get("document_type", "")
                    section_label = chunk_data.get("metadata", {}).get("section", "")
                    st.markdown(f"**`{cid}`** - {source_label} / {section_label}")
                    st.text(chunk_text + ("..." if len(chunk_data.get("text", "")) > 300 else ""))
                    st.markdown("---")

    # Audit trail
    if st.session_state.session_id:
        with st.expander("Audit Trail"):
            audit_data = _api_get(f"/audit/{st.session_state.session_id}")
            if audit_data:
                chain_status = "Valid" if audit_data.get("chain_valid") else "INVALID"
                st.caption(f"Hash chain: {chain_status}")
                for entry in audit_data.get("entries", []):
                    ts = entry.get("timestamp", "")
                    etype = entry.get("event_type", "")
                    st.text(f"  [{ts}] {etype}")
            else:
                st.caption("No audit data available")


def _highlight_citations(text: str) -> str:
    """Replace [N] markers with styled spans."""
    def _replace(match):
        num = match.group(1)
        return f'<span class="citation-marker">[{num}]</span>'

    return re.sub(r"\[(\d+)\]", _replace, text)


# Re-render past messages that are assistant messages with metadata.
# (The loop above calls _render_answer only if the function is defined,
# so we re-render here after the function definition.)
if st.session_state.messages:
    # Clear and re-display to pick up _render_answer definition.
    pass  # Messages already rendered in the loop above (Streamlit re-runs the full script).

# Chat input
user_input = st.chat_input("Ask a clinical question...")

if user_input:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call the backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = _api_post(
                "/query",
                {
                    "patient_id": st.session_state.patient_id,
                    "query": user_input,
                    "session_id": st.session_state.session_id,
                },
            )

        if result:
            answer_msg = {
                "role": "assistant",
                "content": result.get("answer_text", ""),
                "metadata": {
                    "answer_id": result.get("answer_id"),
                    "query_id": result.get("query_id"),
                    "verdict": result.get("verdict"),
                    "confidence": result.get("confidence", 0),
                    "latency_ms": result.get("latency_ms", 0),
                    "abstention_reason": result.get("abstention_reason"),
                    "evidence_pack": result.get("evidence_pack"),
                    "citations": result.get("citations", []),
                },
            }
            st.session_state.messages.append(answer_msg)
            _render_answer(answer_msg)
        else:
            fallback_msg = {
                "role": "assistant",
                "content": "Sorry, I was unable to process your query. Please check the backend logs.",
            }
            st.session_state.messages.append(fallback_msg)
            st.markdown(fallback_msg["content"])
