"""
NeoStats AI Chatbot — Prompt Engineering Utilities
System prompts, response mode modifiers, and chat history management.
"""

import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# ─── Domain System Prompts ─────────────────────────────────────────────────────

DOMAIN_SYSTEM_PROMPTS = {
    "Healthcare": """You are MediBot Pro, an intelligent healthcare information assistant.
Your role is to provide accurate, evidence-based medical information to help users
understand symptoms, conditions, medications, and wellness topics.

CRITICAL GUIDELINES:
- Always recommend consulting a qualified healthcare professional for diagnosis and treatment
- Never replace professional medical advice, diagnosis, or prescriptions
- Clearly distinguish between general health information and medical advice
- Cite sources when discussing specific medical statistics or studies
- Use empathetic, clear language appropriate for non-medical audiences
- Flag any emergency symptoms and direct to emergency services immediately
- Be transparent about uncertainty — say "I'm not sure" when applicable

When RAG context or web search results are provided, integrate them naturally into your response.
Today's date: {date}""",

    "Legal": """You are LexBot, an intelligent legal information assistant.
You help users understand legal concepts, processes, and general information.

CRITICAL GUIDELINES:
- Always clarify that you provide legal information, NOT legal advice
- Recommend consulting a qualified attorney for specific legal matters
- Cite relevant laws, regulations, or precedents when applicable
- Be accurate about jurisdictional differences
- Today's date: {date}""",

    "Finance": """You are FinBot, an intelligent financial information assistant.
You help users understand financial concepts, markets, and personal finance topics.

CRITICAL GUIDELINES:
- Provide educational information, not personalized investment advice
- Recommend consulting a licensed financial advisor for personal decisions
- Disclaimer: Past performance does not guarantee future results
- Today's date: {date}""",

    "Education": """You are EduBot, an intelligent educational assistant.
You help students and learners understand concepts across all subjects.

GUIDELINES:
- Explain concepts clearly with examples appropriate to the user's level
- Encourage critical thinking and deeper learning
- Provide step-by-step explanations for complex topics
- Today's date: {date}""",

    "General": """You are an intelligent AI assistant powered by NeoStats.
You are helpful, accurate, and thoughtful in your responses.

GUIDELINES:
- Provide clear, accurate, and well-structured responses
- Acknowledge uncertainty when you're not sure about something
- Be concise when appropriate, detailed when needed
- Today's date: {date}""",
}


def get_system_prompt(
    domain: str = "General",
    rag_context: str = "",
    web_context: str = "",
    response_mode: str = "Detailed",
) -> str:
    """
    Build the full system prompt incorporating domain, contexts, and response mode.

    Args:
        domain: Use-case domain (Healthcare, Legal, etc.)
        rag_context: Retrieved document context from RAG
        web_context: Live web search results context
        response_mode: "Concise" or "Detailed"

    Returns:
        Complete system prompt string
    """
    base = DOMAIN_SYSTEM_PROMPTS.get(domain, DOMAIN_SYSTEM_PROMPTS["General"])
    base = base.format(date=datetime.now().strftime("%B %d, %Y"))

    # Response mode instruction
    mode_instruction = (
        "\n\nRESPONSE MODE — CONCISE: Keep your reply under 3 sentences. "
        "Be direct and to the point. No elaboration unless critical."
        if response_mode == "Concise"
        else "\n\nRESPONSE MODE — DETAILED: Provide comprehensive, well-structured responses. "
        "Use headings, bullet points, and examples where helpful. "
        "Explain reasoning and include relevant context."
    )
    base += mode_instruction

    # Inject retrieved context
    if rag_context:
        base += f"\n\n{rag_context}"

    if web_context:
        base += f"\n\n{web_context}"

    if rag_context or web_context:
        base += (
            "\n\nIMPORTANT: Prioritize the above context when answering. "
            "Clearly reference sources where applicable. "
            "If the context doesn't fully answer the question, supplement with your knowledge."
        )

    return base


# ─── Chat History ──────────────────────────────────────────────────────────────

def format_chat_history(history: list[dict], max_turns: int = 20) -> list[dict]:
    """
    Trim and format chat history for LLM consumption.

    Args:
        history: List of {role, content} dicts
        max_turns: Max message pairs to retain

    Returns:
        Trimmed list of messages
    """
    # Keep only the last N turns (each turn = user + assistant)
    max_messages = max_turns * 2
    if len(history) > max_messages:
        history = history[-max_messages:]
    return history


def add_message(history: list[dict], role: str, content: str) -> list[dict]:
    """Append a message to the history list."""
    history.append({"role": role, "content": content})
    return history


# ─── Confidence & Source Citation ─────────────────────────────────────────────

def build_source_footer(
    rag_chunks: list[dict],
    web_results: list,
    show_sources: bool = True,
) -> str:
    """
    Build a sources footer to append to responses.

    Args:
        rag_chunks: Retrieved document chunks
        web_results: Web search result objects
        show_sources: Whether to include the footer

    Returns:
        Formatted footer string (may be empty)
    """
    if not show_sources:
        return ""

    parts = []

    if rag_chunks:
        doc_sources = list({c.get("source", "unknown") for c in rag_chunks})
        parts.append("📚 **Knowledge Base:** " + " | ".join(doc_sources))

    if web_results:
        links = [r.url for r in web_results if r.url][:3]
        if links:
            parts.append("🌐 **Web Sources:**\n" + "\n".join(f"- {l}" for l in links))

    if not parts:
        return ""

    return "\n\n---\n" + "\n\n".join(parts)
