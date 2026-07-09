"""Streamlit demo: query the pipeline and visualize the Corrective-RAG trace.

    streamlit run app/ui.py

Retrieval + CRAG trace need no API key; tick "generate answer" only if
ANTHROPIC_API_KEY is set.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from rag_eval.config import RagConfig  # noqa: E402
from rag_eval.generate import generate  # noqa: E402
from rag_eval.pipeline import REFUSAL, RagPipeline  # noqa: E402

st.set_page_config(page_title="RAG Eval Harness", layout="wide")
st.title("RAG Eval Harness — adaptive reranking + Corrective-RAG")


@st.cache_resource
def _pipeline(mode: str, rerank_policy: str, retrieval_mode: str) -> RagPipeline:
    return RagPipeline(
        RagConfig(mode=mode, rerank_policy=rerank_policy, retrieval_mode=retrieval_mode, k=5)
    )


with st.sidebar:
    mode = st.selectbox("Mode", ["crag", "plain"])
    rerank_policy = st.selectbox("Rerank policy", ["auto", "always", "never", "llm"])
    retrieval_mode = st.selectbox("Retrieval", ["dense", "hybrid", "sparse"])
    do_generate = st.checkbox("Generate answer (needs ANTHROPIC_API_KEY)", value=False)

query = st.text_input("Question", "Does the phone support wireless charging?")

if st.button("Run") and query.strip():
    pipe = _pipeline(mode, rerank_policy, retrieval_mode)
    if mode == "crag":
        result = pipe.corrective(query)
        st.subheader("CRAG trace")
        for r in result.trace:
            corr = f" → {r.correction}" if r.correction else ""
            st.markdown(
                f"**round {r.round_idx}** · `{r.action}` · "
                f"{'reranked' if r.reranked else 'dense'} · {r.n_candidates} cand · "
                f"kept {len(r.kept_chunk_ids)}{corr}  \n_query_: `{r.query}`"
            )
        contexts = result.contexts
        if result.refused:
            st.warning("No relevant chunks — the system would refuse to answer.")
        if do_generate:
            ans = REFUSAL if result.refused else generate(query, contexts, pipe.config)
            st.subheader("Answer")
            st.write(ans)
    else:
        contexts = pipe.retrieve(query)
        if do_generate:
            st.subheader("Answer")
            st.write(pipe.answer(query).answer)

    st.subheader("Retrieved chunks")
    for i, c in enumerate(contexts, start=1):
        st.markdown(f"**[S{i}]** score={c.score:.3f} · sources={c.source_ids}")
        st.caption(c.text[:300] + ("…" if len(c.text) > 300 else ""))
