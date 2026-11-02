"""
Unit tests for answer generation, Ollama interaction, and chatbot input handling.
"""

import pytest
from unittest.mock import MagicMock, patch
from src.generation.answer_generator import AnswerGenerator
from src.chatbot.chat import RAGChatbot


def test_answer_generation_fallback_on_empty_context():
    """Test clear fallback message when no relevant chunks are found."""
    gen = AnswerGenerator(model_name="mock-model")
    ans = gen.generate("What is the policy?", retrieved_chunks=[])
    assert "couldn't find sufficiently relevant information" in ans


def test_answer_generation_fallback_on_empty_question():
    """Test fallback when question is blank."""
    gen = AnswerGenerator(model_name="mock-model")
    ans = gen.generate("   ", retrieved_chunks=[{"text": "Sample text"}])
    assert "Please provide a valid question." in ans


def test_answer_generation_with_mock_ollama():
    """Test successful LLM generation using mocked Ollama response."""
    with patch("src.generation.answer_generator.ollama.chat") as mock_chat:
        mock_chat.return_value = {
            "message": {
                "content": "The company mission is to continue leading nurse triage."
            }
        }

        gen = AnswerGenerator(model_name="qwen2.5:1.5b")
        chunks = [{"text": "Our mission is to continue to lead the field of nurse triage."}]
        answer = gen.generate("What is the mission?", chunks)

        assert "leading nurse triage" in answer
        mock_chat.assert_called_once()


def test_ollama_connection_error_handling():
    """Test graceful user-facing error message when Ollama service is unreachable."""
    with patch("src.generation.answer_generator.ollama.chat", side_effect=ConnectionError("Failed to connect")):
        gen = AnswerGenerator(model_name="qwen2.5:1.5b")
        chunks = [{"text": "Document context here."}]
        answer = gen.generate("What is the policy?", chunks)

        assert "Unable to connect to Ollama" in answer


def test_chatbot_vague_question_handling():
    """Test chatbot asks for clarification on vague single-word question."""
    mock_retriever = MagicMock()
    mock_generator = MagicMock()

    chatbot = RAGChatbot(
        retriever=mock_retriever,
        answer_generator=mock_generator,
        chunks=["Some chunk"],
        metadata=[{"page_number": 1}]
    )

    response = chatbot.ask("employees?")
    assert response["retrieval_mode"] == "clarification"
    assert "Could you please be more specific?" in response["answer"]
    # Ensure retriever was not called unnecessarily
    mock_retriever.retrieve.assert_not_called()


def test_chatbot_exact_role_lookup():
    """Test chatbot performs exact lookup for CEO/COO names without calling LLM."""
    mock_retriever = MagicMock()
    mock_generator = MagicMock()

    chunks = [
        "Advisory Board Members:\nCharu Raheja, PhD - Chair and CEO\nPhD in Finance",
        "Advisory Board Members:\nRavi Raheja, MD - Medical Director and COO"
    ]
    metadata = [{"page_number": 6}, {"page_number": 6}]

    chatbot = RAGChatbot(
        retriever=mock_retriever,
        answer_generator=mock_generator,
        chunks=chunks,
        metadata=metadata
    )

    # 1. Who is the CEO?
    resp_ceo = chatbot.ask("Who is the CEO?")
    assert resp_ceo["retrieval_mode"] == "exact_lookup"
    assert "Charu Raheja" in resp_ceo["answer"]

    # 2. Name of CEO
    resp_ceo_name = chatbot.ask("Name of CEO.")
    assert resp_ceo_name["retrieval_mode"] == "exact_lookup"
    assert "Charu Raheja" in resp_ceo_name["answer"]

    # 3. Who is the COO?
    resp_coo = chatbot.ask("Who is the COO?")
    assert resp_coo["retrieval_mode"] == "exact_lookup"
    assert "Ravi Raheja" in resp_coo["answer"]

    # 4. Name of COO
    resp_coo_name = chatbot.ask("Name of COO.")
    assert resp_coo_name["retrieval_mode"] == "exact_lookup"
    assert "Ravi Raheja" in resp_coo_name["answer"]


def test_ceo_coo_salary_intent_does_not_trigger_name_lookup():
    """
    Test that queries asking for CEO or COO salary DO NOT trigger
    the exact person name lookup and instead route to the RAG pipeline.
    """
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        {"chunk_id": 0, "score": 1.5, "metadata": {"page_number": 71}, "text": "Salaries are strictly confidential."}
    ]
    mock_generator = MagicMock()
    mock_generator.generate.return_value = "The document states that salaries are confidential."

    chunks = [
        "Charu Raheja, PhD - Chair and CEO",
        "Ravi Raheja, MD - Medical Director and COO"
    ]
    metadata = [{"page_number": 6}, {"page_number": 6}]

    chatbot = RAGChatbot(
        retriever=mock_retriever,
        answer_generator=mock_generator,
        chunks=chunks,
        metadata=metadata
    )

    # Query: 'salary of ceo'
    resp1 = chatbot.ask("salary of ceo")
    assert resp1["retrieval_mode"] == "hybrid_rag"
    assert "Charu Raheja, PhD" not in resp1["answer"]
    mock_retriever.retrieve.assert_called()

    # Query: 'What is the CEO salary?'
    resp2 = chatbot.ask("What is the CEO salary?")
    assert resp2["retrieval_mode"] == "hybrid_rag"

    # Query: 'What is the COO salary?'
    resp3 = chatbot.ask("What is the COO salary?")
    assert resp3["retrieval_mode"] == "hybrid_rag"


def test_vague_query_employ_and_tailored_suggestions():
    """Test that single vague words like 'employ' or 'leave' trigger tailored clarification."""
    mock_retriever = MagicMock()
    mock_generator = MagicMock()

    chatbot = RAGChatbot(
        retriever=mock_retriever,
        answer_generator=mock_generator,
        chunks=["Some text"],
        metadata=[{"page_number": 1}]
    )

    # Test 'employ'
    resp_employ = chatbot.ask("employ")
    assert resp_employ["retrieval_mode"] == "clarification"
    assert "Employee responsibilities" in resp_employ["answer"]
    assert "Leave policy" in resp_employ["answer"]

    # Test 'leave'
    resp_leave = chatbot.ask("leave")
    assert resp_leave["retrieval_mode"] == "clarification"
    assert "leave policy" in resp_leave["answer"]


def test_missing_leave_quantity_and_manager_prompt_rules():
    """Test that AnswerGenerator system prompt contains explicit rules for leave quantities and manager names."""
    gen = AnswerGenerator(model_name="mock-model")
    prompt = gen.system_prompt

    # Ensure rule against inventing leave counts exists
    assert "NEVER invent leave numbers" in prompt or "never invent" in prompt.lower()
    # Ensure rule against fabricating manager names exists
    assert "NEVER fabricate personal names" in prompt or "never fabricate" in prompt.lower()
    # Ensure rule against inventing salary figures exists
    assert "NEVER invent salary figures" in prompt or "never invent salary" in prompt.lower()

