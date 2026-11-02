"""
Answer Generator module.

Uses local Ollama LLM (qwen2.5:1.5b) to synthesize concise, strictly grounded
answers based only on retrieved context passages.
"""

from typing import Optional
import logging
import ollama

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """You are a professional, document-grounded AI assistant for company policies.

Your task is to answer the user's question using ONLY the provided DOCUMENT CONTEXT.

STRICT RULES:
1. Grounding: Answer strictly using facts directly stated in the context. Do not invent, extrapolate, or assume any information outside the provided excerpts.
2. Insufficient Information: If the provided context does not contain enough information to answer the question, clearly state:
   "The provided document does not contain sufficient information to answer this question."
3. Leave & Vacation Policies:
   - When answering questions about leave, vacation, or calling in sick, explain ONLY the procedures directly governing leave (e.g., calling in sick, using leave, requesting a change of duty, notifying a supervisor, or notifying a company doctor to avoid unsafe workplace practices).
   - NEVER include unrelated rules such as prescription medication guidelines, drug policies, or salary confidentiality in leave answers, even if they appear in the same excerpt.
   - If the user asks how many leaves can be taken or for specific quantities, explicitly state that the document does not specify the exact number of leaves or vacation days allowed. NEVER invent leave numbers.
4. Salaries & Compensation:
   - If asked about the salary, pay, or compensation of the CEO, COO, or employees, explain that individual salaries and rates of pay are determined on an individual basis, are considered strictly confidential, and specific salary numbers are not provided in the document.
   - NEVER invent salary figures.
5. Manager Names & Positions:
   - If asked for names of managers: State that the document mentions managerial titles/roles (such as Nurse Managers, Client Relations Manager, Sales & Accounts Manager, Marketing Manager, IT Relations Manager, and Clinical Manager), but individual employee names for these manager positions are not specified in the document.
   - NEVER fabricate personal names.
6. Precision: Keep your answer direct, professional, factual, and concise. Preserve exact names, titles, and procedures as written.
"""


class AnswerGenerator:
    """
    Interfaces with Ollama to generate context-grounded responses.
    """

    def __init__(
        self,
        model_name: str = "qwen2.5:1.5b",
        temperature: float = 0.0,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT
    ) -> None:
        """
        Initialize the Answer Generator.

        Args:
            model_name: Ollama model identifier.
            temperature: LLM sampling temperature (0.0 for deterministic answers).
            system_prompt: System prompt instructing the model.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.system_prompt = system_prompt

    def generate(
        self,
        user_question: str,
        retrieved_chunks: list[dict]
    ) -> str:
        """
        Generate an answer to the user question using retrieved chunk contexts.

        Args:
            user_question: User's original or corrected question.
            retrieved_chunks: List of retrieved chunk dictionaries containing 'text'.

        Returns:
            Generated answer string or a clear fallback message.
        """
        if not user_question or not user_question.strip():
            return "Please provide a valid question."

        if not retrieved_chunks:
            return (
                "I couldn't find sufficiently relevant information "
                "in the document to answer that question."
            )

        # Concatenate retrieved chunk texts with clear separators
        context_sections = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            text = chunk.get("text", "").strip()
            page = chunk.get("metadata", {}).get("page_number", "unknown")
            if text:
                context_sections.append(f"[Excerpt {i} - Page {page}]:\n{text}")

        if not context_sections:
            return (
                "I couldn't find sufficiently relevant information "
                "in the document to answer that question."
            )

        combined_context = "\n\n".join(context_sections)

        user_prompt = f"""DOCUMENT CONTEXT:
{combined_context}

QUESTION:
{user_question}

Answer based ONLY on the document context above."""

        try:
            logger.info(f"Generating answer using Ollama model: {self.model_name}...")
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                options={"temperature": self.temperature}
            )

            answer = response.get("message", {}).get("content", "").strip()
            if not answer:
                return "The model did not return an answer. Please rephrase your question."

            return answer

        except ConnectionError as conn_err:
            logger.error(f"Cannot connect to Ollama: {conn_err}")
            return (
                "Unable to connect to Ollama. Please make sure the Ollama application "
                "or 'ollama serve' is running locally on your computer."
            )
        except Exception as err:
            logger.error(f"Error during LLM answer generation: {err}", exc_info=True)
            return (
                f"An error occurred while generating the answer: {str(err)}. "
                "Please verify that Ollama is running and the model is downloaded."
            )
