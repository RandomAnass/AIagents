# Autonomous Research Assistant 🤖📚  
> Rapidly discover, rank, and explore research papers — powered by LLM agents.

![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red)
![LangGraph](https://img.shields.io/badge/LangGraph-0.0.32-green)
![WikiRace](https://img.shields.io/badge/WikiRace-1.0.0-yellow)

---

## 🚀 Quickstart

```bash
pip install -r requirements.txt
streamlit run App.py
```
 

---

##  Features

| Module | Description |
|--------|-------------|
| **Autonomous Research Assistant** | Search arXiv, summarize papers, score relevance, veto low-quality ones via Critic agent. |
| **LangGraph DAG** | `Retrieve → Summarise → Score → Critic → Rank` nodes orchestrated with stateful graph execution. |
| **LLMOps** | Robust back-off, retries, token-efficient prompting (JSON schema for parsing). |
| **WikiRace Mini-game** | Human vs LLM vs Oracle (BFS) race across Wikipedia links. Shows agent reasoning . |



---

## Core Technologies

- **LangGraph** — modular DAG orchestration for LLM agent flows
- **OpenRouter** — LLM API for multiple LLM testing 
- **arXiv API** — lightweight paper retrieval
- **Streamlit** — interactive frontend (search, chat, quiz)
- **Wikipedia API** — Wikirace graph exploration

---

## Future Enhancements

- Add **vector-based RAG** retrieval (for full papers, not just abstracts)
- Enable **LangGraph memory** for multi-turn conversation agents
- Fix bugs + add more tests


