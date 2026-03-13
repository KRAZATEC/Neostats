# 🩺 MediBot Pro — NeoStats AI Chatbot

> **Intelligent Healthcare Assistant** powered by RAG, Live Web Search, and Multi-LLM support.  
> Built for the NeoStats AI Engineer Case Study Challenge.

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

---

## 🎯 Use Case

**MediBot Pro** solves a real-world healthcare information gap: patients and healthcare students often struggle to find accurate, evidence-based medical information quickly. MediBot Pro acts as an intelligent medical assistant that:

- Answers questions using **your own uploaded medical documents** (RAG)
- Retrieves **live, up-to-date medical research** via web search
- Adapts response depth with **Concise / Detailed modes**
- Supports **three leading LLM providers** for flexibility

---

## ✨ Features

### Mandatory Features
| Feature | Status | Implementation |
|--------|--------|----------------|
| RAG Integration | ✅ | FAISS + SentenceTransformers |
| Live Web Search | ✅ | Serper / Tavily / Brave APIs |
| Concise vs Detailed | ✅ | Sidebar radio toggle |

### Bonus / Unique Features
| Feature | Description |
|--------|-------------|
| 🏥 Multi-Domain Support | Healthcare, Legal, Finance, Education, General |
| ⚡ Streaming Responses | Token-by-token streaming for all providers |
| 📊 Session Analytics | Messages, response time, RAG hits, web searches |
| 📥 Chat Export | Download conversation as TXT or JSON |
| 🌡️ Temperature Control | Fine-tune LLM creativity per session |
| 📂 Multi-file Upload | PDF, DOCX, TXT, CSV, MD support |
| 🔗 Source Citations | Transparent RAG + web result attribution |
| 🎨 Custom Dark UI | Medical-themed design with animations |
| 🔌 Provider Fallback | Serper → Tavily → Brave search chain |
| 🧹 Knowledge Base Reset | Clear indexed documents anytime |

---

## 🏗️ Architecture

```
project/
├── config/
│   ├── __init__.py
│   └── config.py          ← All API keys, model settings, RAG/search config
│
├── models/
│   ├── __init__.py
│   ├── llm.py             ← OpenAI / Groq / Gemini unified interface
│   └── embeddings.py      ← SentenceTransformer + FAISS vector store
│
├── utils/
│   ├── __init__.py
│   ├── document_processor.py  ← PDF/DOCX/CSV extraction, chunking, RAG pipeline
│   ├── web_search.py          ← Serper / Tavily / Brave search integration
│   ├── prompt_utils.py        ← System prompts, response modes, history management
│   └── analytics.py           ← Session tracking, chat export
│
├── data/
│   ├── knowledge_base/    ← Sample documents
│   └── vector_store/      ← FAISS index (auto-generated)
│
├── .streamlit/
│   ├── config.toml        ← Theme + server settings
│   └── secrets.toml.example
│
├── app.py                 ← Main Streamlit UI
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/neostats-medibot-pro.git
cd neostats-medibot-pro

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml with your API keys
```

Or set environment variables:
```bash
export GROQ_API_KEY="your-groq-key"        # Free at console.groq.com
export OPENAI_API_KEY="your-openai-key"    # Optional
export GEMINI_API_KEY="your-gemini-key"    # Optional
export SERPER_API_KEY="your-serper-key"    # Free tier at serper.dev
```

### 3. Run

```bash
streamlit run app.py
```

Visit `http://localhost:8501`

---

## 🔑 Getting Free API Keys

| Provider | Free Tier | Link |
|----------|-----------|------|
| Groq | ✅ Generous free tier | [console.groq.com](https://console.groq.com) |
| Gemini | ✅ Free tier available | [aistudio.google.com](https://aistudio.google.com) |
| Serper | ✅ 2,500 free searches | [serper.dev](https://serper.dev) |
| Tavily | ✅ 1,000 free searches | [tavily.com](https://tavily.com) |

---

## 📖 Usage Guide

### Chatting
1. Connect a language model using the sidebar
2. Type your healthcare question in the chat input
3. Toggle RAG (if documents uploaded) and Web Search as needed

### RAG Setup
1. Enable "Knowledge Base (RAG)" in sidebar
2. Upload PDF/DOCX/TXT medical documents
3. The chatbot will automatically retrieve relevant passages

### Web Search
1. Enable "Live Web Search" in sidebar
2. Enter a Serper or Tavily API key
3. Queries containing keywords like "latest", "current", "research" trigger searches

---

## 🛡️ Medical Disclaimer

MediBot Pro provides **general health information only** and is **not a substitute for professional medical advice**. Always consult a qualified healthcare professional for diagnosis, treatment, or medical decisions. In emergencies, call your local emergency services immediately.

---

## 📄 License

MIT License — built for educational purposes as part of the NeoStats AI Engineer Challenge.
