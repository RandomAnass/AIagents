"""
Limitation: 1 500 requests / day, 15 req/min, so in our case UI caps results to 1 – 5 (default 3)
each paper costs 3 requests (summary, score, critic).
* Token budget:
even with long abstracts (≈ 400 input tokens + ≈ 100 output tokens per call):
500 tokens × 15 calls  ≈  7 500 tokens/search
That’s less than 1 % of the 1 000 000 tokens/minute ceiling.  
we rate-limited by requests, not by tokens. we can add more tokens (pdf, chat, quiz etc)
"""
from __future__ import annotations
import streamlit as st
import pandas as pd
import arxiv, matplotlib.pyplot as plt, networkx as nx
from langgraph.graph import StateGraph, END

import json, textwrap, time, random, requests, os
from typing import TypedDict, Any

# Ui
st.set_page_config(
    page_title="Siemens Research Assistant",
    page_icon="🧑‍🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

test_CSS = """
<style>
h1, h2 {border-bottom: 2px solid #009999; padding-bottom:4px;}
label, .stSelectbox label {font-weight:600;}
</style>
"""
st.markdown(test_CSS, unsafe_allow_html=True)


# Sidebar for key, not practical, just for demo

with st.sidebar:
    st.header("🔑 Gemini API Key")
    gemini_key = st.text_input("Paste your key", type="password",
                               placeholder="AIza…")
    if not gemini_key:
        st.warning("Enter your Gemini key to enable the assistant.", icon="⚠️")

###############################################################################
#   gemini wrapper (simple REST – no client lib)/ you can check the notebook for openrouter option
# openrouter uses the openai API 
###############################################################################
GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash:generateContent?key={key}"
)


def call_llm(prompt: str, key: str, *, temperature: float = 0.2,
             retries: int = 3, backoff: float = 2.0) -> str:
    """call to Gemini """
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature},
    }
    url = GEMINI_ENDPOINT.format(key=key)

    for attempt in range(retries):
        try:
            resp = requests.post(url, json=payload, timeout=30)
            if resp.status_code == 200:
                data: Any = resp.json()
                print(data, "gemini response")
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            else:
                #429 (rate) or 400 (key)
                print('444444!!!!')
                msg = f"Gemini error {resp.status_code}: {resp.text}"
                if attempt == retries - 1:
                    raise RuntimeError(msg)
        except (requests.RequestException, RuntimeError) as e:
            if attempt == retries - 1:
                raise
        # exponential back‑off with jitter
        wait = backoff * (2 ** attempt) + random.random()
        time.sleep(wait)

###############################################################################
# 3 · LangGraph  
###############################################################################

# pipline / similar to a Medium  article i read
class PaperState(TypedDict, total=False):
    query: str
    max_k: int
    papers: pd.DataFrame
    result: pd.DataFrame
    _key: str  


def build_pipeline() -> StateGraph:
    graph = StateGraph(PaperState)

 
    def retrieve(state: PaperState) -> PaperState:
        q, k = state["query"], state["max_k"]
        search = arxiv.Search(query=q, max_results=k,
                              sort_by=arxiv.SortCriterion.Relevance) # we can change the sort_by to date or relevance
        # search = arxiv.Search(query=q, max_results=k, sort_by=arxiv.SortCriterion.Date) 
        print("arxiv result", search.results())
        rows = [{
            "title": r.title,
            "abstract": r.summary,
            "authors": ", ".join(a.name for a in r.authors),
            "url": r.entry_id,
        } for r in search.results()]
        papers= pd.DataFrame(rows)
        print(papers) # TODO remove all prints
        return {"papers": papers}

    def summarise(state: PaperState) -> PaperState:
        df = state["papers"].copy()
        key = state["_key"]
        df["summary"] = df["abstract"].apply(
            lambda abs_: call_llm(
                f"Summarise the following abstract in two sentences.\n\n{abs_}",
                key,
            ) # directly using LLM with apply might be tricky
        )
        return {"papers": df}

    def score(state: PaperState) -> PaperState:
        df, topic, key = state["papers"].copy(), state["query"], state["_key"]
        scores, reasons = [], []
        for abs_ in df["abstract"]:
            prompt = textwrap.dedent(f"""
                Topic: '{topic}'
                Return JSON like {{"score": 87, "why": "…"}} where score is 0‑100, score is how relevant the paper is to the query.
                Abstract:\n{abs_}
            """)
            
            try:
                #print(call_llm(prompt, key, temperature=0))
                #js = json.loads(call_llm(prompt, key, temperature=0))
                #print(js)
                response = call_llm(prompt, key, temperature=0)
                if response.startswith("```json"):
                    print("sss")
                    response = str(response)[8:-4] # Remove the ```json and ending ```
                #print(2)
                #print(response)
                #response = json.loads(response) 
                response = json.loads(response) 
                #print(response)
                scores.append(int(response.get("score", 0)))
                reasons.append(response.get("why", "—"))
            except Exception as e:
                print("parse error", str(e))
                scores.append(0)
                reasons.append("parse‑error")
        df["score"], df["reason"] = scores, reasons
        return {"papers": df}

    def critic(state: PaperState) -> PaperState:
        df, topic, key = state["papers"].copy(), state["query"], state["_key"]
        keep, critic_reason = [], []
        for s, abs_ in zip(df["score"], df["abstract"]):
            if s >= 80:
                keep.append(True); critic_reason.append("score 80")
            else:
                prompt = (
                    f"Topic: {topic}\nScore so far: {s}\n"
                    "Should we keep this paper, is it relevant enough to the topic? Answer yes/no and one reason.\n" + abs_
                )
                reply = call_llm(prompt, key, temperature=0)
                keep.append("yes" in reply.lower())
                critic_reason.append(reply.strip())
        df["keep"], df["critic_reason"] = keep, critic_reason
        print(df, 'after critic')
        return {"papers": df}

    def rank(state: PaperState) -> PaperState:
        df = (state["papers"][state["papers"]["keep"]]
              .sort_values("score", ascending=False)
              .reset_index(drop=True))
        return {"result": df}

    #   graph edges  
    graph.add_node("Retrieve", retrieve)
    graph.add_node("Summarise", summarise)
    graph.add_node("Score", score)
    graph.add_node("Critic", critic)
    graph.add_node("Rank", rank)

    graph.add_edge("Retrieve", "Summarise")
    graph.add_edge("Summarise", "Score")
    graph.add_edge("Score", "Critic")
    graph.add_edge("Critic", "Rank")
    graph.add_edge("Rank", END)
    graph.set_entry_point("Retrieve")
    return graph.compile()

PIPELINE = build_pipeline()

###############################################################################
# 4 · Streamlit tabs
###############################################################################
st.title("🧑‍🔬 Siemens Autonomous Research Assistant ")

tabs = st.tabs(["🔎 Search & Rank", "🧠 Pipeline", "💬 Paper Chat", "📝 Quiz Me"])

# -------------- TAB 1 — Search & Rank --------------
with tabs[0]:
    st.header("Search arXiv & rank papers")
    query = st.text_input("Research topic", value="multi‑agent reinforcement learning")
    k = st.slider("Number of results", min_value=1, max_value=5, value=3,
                  help="arXiv results to fetch (max 5 to respect Gemini rate limits)")
    run = st.button("🚀 Run")

    if run and gemini_key:
        with st.spinner("Running pipeline …"):
            state_in: PaperState = {
                "query": query,
                "max_k": k,
                "_key": gemini_key,
            }
            out = PIPELINE.invoke(state_in)
        df = out["result"]
        st.session_state["papers_df"] = df

        st.subheader("Ranked papers")
        st.dataframe(df[["title", "score", "critic_reason"]], height=300)

        # small bar chart
        top = df.head(min(10, len(df)))
        if not top.empty:
            fig, ax = plt.subplots(figsize=(8, 0.6 * len(top) + 1))
            ax.barh(top["title"], top["score"])
            ax.invert_yaxis()
            ax.set_xlabel("Score")
            ax.set_title("Top relevance scores")
            st.pyplot(fig)

# -------------- TAB 2 — Pipeline details --------------
#with tabs[1]:
#    st.header("Gemini 2.0 Flash pipeline")
#    st.markdown("Gemini endpoint: **gemini-2.0-flash:generateContent**   ·   3 calls/paper")
#    with st.expander("Show LangGraph DAG"):
#        G = nx.DiGraph([("Retrieve", "Summarise"),
#                        ("Summarise", "Score"),
#                        ("Score", "Critic"),
#                        ("Critic", "Rank")])
#        fig, ax = plt.subplots(figsize=(5, 2.5))
#        nx.draw_networkx(G, arrows=True, node_color="#009999",
#                         node_size=2000, font_color="white", ax=ax)
#        ax.axis("off"); st.pyplot(fig)
with tabs[1]:
    with st.expander("Show LangGraph DAG"):
        G = nx.DiGraph([
            ("Retrieve", "Summarise"),
            ("Summarise", "Score"),
            ("Score", "Critic"),
            ("Critic", "Rank"),
        ])

        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot")  
        except:
            pos = nx.spring_layout(G)  

        fig, ax = plt.subplots(figsize=(4, 5))
        nx.draw(G, pos,
                with_labels=True,
                arrows=True,
                node_color="#009999",
                node_size=2500,
                font_color="white",
                font_weight="bold",
                ax=ax)
        ax.set_title("LangGraph Pipeline", fontsize=12)
        ax.axis("off")
        st.pyplot(fig)

# -------------- TAB 3 — Chat with paper --------------
with tabs[2]:
    st.header("Chat with a selected paper")
    if "papers_df" not in st.session_state:
        st.info("Run a search first in the *Search & Rank* tab.")
    else:
        df = st.session_state["papers_df"]
        sel = st.selectbox("Choose paper", df["title"].tolist())
        paper = df[df["title"] == sel].iloc[0]
        st.markdown("##### Abstract")
        st.write(paper["abstract"])

        if "chat_hist" not in st.session_state:
            st.session_state.chat_hist = []

        user_q = st.text_input("Ask something about the paper")
        if st.button("Send") and user_q and gemini_key:
            history = "\n".join(
                f"User: {h['q']}\nAssistant: {h['a']}" for h in st.session_state.chat_hist
            )
            prompt = (
                f"{history}\nUser: {user_q}\nAssistant:\n\n[Paper abstract below]\n{paper['abstract']}"
            )
            answer = call_llm(prompt, gemini_key)
            st.session_state.chat_hist.append({"q": user_q, "a": answer})

        for h in st.session_state.chat_hist[::-1]:
            st.markdown(f"**You:** {h['q']}")
            st.markdown(f"**Assistant:** {h['a']}")

# -------------- TAB 4 — Quiz generator --------------
with tabs[3]:
    st.header("Generate a quiz from a paper")
    if "papers_df" not in st.session_state:
        st.info("Run a search first.")
    else:
        df = st.session_state["papers_df"]
        sel = st.selectbox("Paper", df["title"].tolist(), key="quiz_paper")
        paper = df[df["title"] == sel].iloc[0]

    if st.button("Generate quiz") and gemini_key:
        prompt = textwrap.dedent(f"""
            You are a JSON generator ONLY.
            Output MUST be a valid JSON array of exactly 3 objects and nothing else.

            Format of each object:
            {{
                "question": "string",
                "options": ["A ...", "B ...", "C ...", "D ..."],
                "correct": "A"
            }}

            Abstract:
            {paper['abstract']}
        """)

        raw = call_llm(prompt, gemini_key)        # <-- your Gemini helper

        try:
            # -------- strip optional ``` fences ----------
            cleaned = raw.strip()
            for fence in ("```json", "```"):
                if cleaned.startswith(fence):
                    cleaned = cleaned[len(fence):]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]

            quiz = json.loads(cleaned)
            st.session_state.quiz = quiz

        except Exception as e:
            st.error(f"Failed to parse quiz ({e})")
            st.subheader("Raw model output:")
            st.code(raw, language="markdown")
            st.stop()      # stop rendering the rest of the tab

        if "quiz" in st.session_state:
            answers = []
            for i, q in enumerate(st.session_state.quiz):
                st.subheader(f"Q{i+1}: {q['question']}")
                ans = st.radio("Your answer", q["options"], key=f"ans_{i}")
                answers.append(ans.split()[0] == q["correct"])  # compare letter

            if st.button("Submit answers"):
                score = sum(answers)
                st.success(f"Score: {score} / {len(answers)}")
                for ok, q in zip(answers, st.session_state.quiz):
                    st.write("✅" if ok else "❌",
                             f"Correct: **{q['correct']}** – {q['question']}")
