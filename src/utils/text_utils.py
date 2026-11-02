"""
Text processing and query assistance utilities.

Contains helper functions for text cleaning, tokenization, vocabulary building,
spelling correction, query expansion, vague question detection, and exact executive role extraction.
"""

import difflib
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Common English question words that should not be corrected or filtered out unnecessarily
COMMON_QUERY_WORDS = {
    "what", "which", "who", "whom", "whose", "where", "when",
    "why", "how", "is", "are", "am", "was", "were", "do",
    "does", "did", "the", "a", "an", "of", "in", "on", "at",
    "to", "for", "from", "by", "with", "about", "and", "or",
    "but", "not", "this", "that", "these", "those", "it",
    "its", "company", "tell", "me", "please", "can", "could",
    "would", "should", "i", "we", "you", "they", "their",
    "my", "our", "your", "name", "names", "give", "explain",
    "list", "show"
}

# Domain vocabulary for HR and healthcare company policies
DOMAIN_VOCABULARY = {
    "employ", "employee", "employees", "employer", "employers",
    "staff", "personnel", "workforce", "worker", "workers",
    "benefit", "benefits", "responsibility", "responsibilities",
    "policy", "policies", "procedure", "procedures",
    "salary", "salaries", "wage", "wages", "leave",
    "vacation", "holiday", "training", "attendance",
    "performance", "conduct", "termination", "recruitment",
    "hiring", "onboarding", "manager", "management",
    "mission", "purpose", "nurse", "triage", "ceo", "coo"
}

# Common stop words for BM25 search
STOP_WORDS = {
    "what", "is", "the", "a", "an", "of", "in", "to", "and", "for", "on", "at", "by", "with"
}


def clean_text(text: str) -> str:
    """Normalize whitespace and strip leading/trailing spaces."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str, remove_stopwords: bool = False) -> list[str]:
    """
    Extract lowercase alphanumeric tokens from text.
    Optionally filter out common stopwords.
    """
    if not text:
        return []
    tokens = re.findall(r"\b\w+\b", text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    return tokens


def build_vocabulary(chunks: list[str]) -> set[str]:
    """
    Build a comprehensive vocabulary from document chunks, common query words,
    and domain terms. Used for conservative spelling correction.
    """
    document_words: set[str] = set()
    for chunk in chunks:
        words = re.findall(r"\b[a-zA-Z]+\b", chunk.lower())
        document_words.update(words)

    return document_words | COMMON_QUERY_WORDS | DOMAIN_VOCABULARY


def correct_spelling(user_question: str, vocabulary: set[str]) -> str:
    """
    Conservatively correct obvious spelling mistakes using the vocabulary.
    Words with fewer than 4 letters or words already in the vocabulary are untouched.
    """
    if not user_question:
        return ""

    def replace_word(match: re.Match) -> str:
        original = match.group(0)
        lower = original.lower()

        # Skip short words and known words
        if len(lower) < 4 or lower in vocabulary:
            return original

        matches = difflib.get_close_matches(lower, vocabulary, n=1, cutoff=0.80)
        if not matches:
            return original

        corrected = matches[0]
        # Preserve original casing pattern
        if original.isupper():
            return corrected.upper()
        if original.istitle():
            return corrected.title()
        return corrected

    return re.sub(r"[A-Za-z]+", replace_word, user_question)


def expand_query(user_question: str) -> str:
    """
    Expand user queries with relevant domain synonyms to improve retrieval recall.
    For example: 'mission' is expanded with 'company mission purpose nurse triage services'.
    """
    if not user_question:
        return ""

    question_lower = user_question.lower()
    expansions: list[str] = [user_question]

    if any(term in question_lower for term in ["employee", "staff", "personnel", "workforce"]):
        expansions.append("employee employees staff personnel workforce workers team members")

    if "benefit" in question_lower:
        expansions.append("benefits compensation insurance employee programs")

    if any(term in question_lower for term in ["responsibility", "responsibilities", "duties"]):
        expansions.append("job duties roles expectations tasks employee responsibilities")

    if "mission" in question_lower:
        expansions.append("company mission purpose nurse triage services")

    if "leave" in question_lower or "vacation" in question_lower or "absence" in question_lower or "sick" in question_lower:
        expansions.append("leave policy sick leave time off absence vacation call in sick personnel procedures")

    if "manager" in question_lower or "managers" in question_lower:
        expansions.append("manager managers nurse manager client relations manager sales account manager marketing manager IT relations manager clinical manager leadership")

    if "salary" in question_lower or "compensation" in question_lower or "pay" in question_lower or "wage" in question_lower:
        expansions.append("salaries rates of pay confidential employee compensation individual salary wages public information")

    if "ceo" in question_lower and not any(term in question_lower for term in ["salary", "pay", "compensation"]):
        expansions.append("chief executive officer chair Charu Raheja")

    if "coo" in question_lower and not any(term in question_lower for term in ["salary", "pay", "compensation"]):
        expansions.append("chief operating officer Ravi Raheja medical director")

    return " ".join(expansions)


VAGUE_TERMS_MAP = {
    "employ": "Employee responsibilities, Employee benefits, Leave policy, Employee conduct, or Training policies",
    "employee": "Employee responsibilities, Employee benefits, Leave policy, Employee conduct, or Training policies",
    "employees": "Employee responsibilities, Employee benefits, Leave policy, Employee conduct, or Training policies",
    "employer": "Employee responsibilities, Employee benefits, Leave policy, Employee conduct, or Training policies",
    "employers": "Employee responsibilities, Employee benefits, Leave policy, Employee conduct, or Training policies",
    "staff": "staff classifications, job responsibilities, or conduct policies",
    "personnel": "personnel procedures, classifications, or responsibilities",
    "workforce": "workforce classifications or responsibilities",
    "worker": "worker classifications, responsibilities, or conduct",
    "workers": "worker classifications, responsibilities, or conduct",
    "policy": "general company policies, drug policy, harassment policy, or data security policy",
    "policies": "general company policies, drug policy, harassment policy, or data security policy",
    "leave": "leave policy, calling in sick, or absence procedures",
    "leaves": "leave policy, calling in sick, or absence procedures",
    "vacation": "vacation guidelines, calling in sick, or leave policies",
    "salary": "salary confidentiality rules or employee classifications",
    "salaries": "salary confidentiality rules or employee classifications",
    "pay": "salary confidentiality rules or employee classifications",
    "benefit": "employee benefits and employee classifications",
    "benefits": "employee benefits and employee classifications",
    "manager": "manager roles, nurse managers, client relations managers, or leadership positions",
    "managers": "manager roles, nurse managers, client relations managers, or leadership positions",
}


def is_vague_question(user_question: str) -> bool:
    """
    Detect whether a question is simply a single broad word or topic (e.g. 'employ' or 'leave?'),
    where asking for clarification provides a better user experience than guessing.
    """
    if not user_question:
        return False

    normalized = re.sub(r"[^a-zA-Z\s]", "", user_question.lower()).strip()
    return normalized in VAGUE_TERMS_MAP


def get_vague_question_suggestion(user_question: str) -> str:
    """
    Return a tailored clarification prompt for a vague keyword query.
    """
    normalized = re.sub(r"[^a-zA-Z\s]", "", user_question.lower()).strip()
    topic_suggestions = VAGUE_TERMS_MAP.get(
        normalized,
        "employee responsibilities, company mission, leave policies, or leadership"
    )
    return (
        f"Could you please be more specific? For example, you can ask about: "
        f"{topic_suggestions}."
    )


# Keywords indicating the user is asking for non-name attributes
NON_NAME_INTENTS = {
    "salary", "compensation", "pay", "wage", "wages", "bonus", "earnings",
    "income", "money", "age", "phone", "number", "email", "address",
    "contact", "responsibility", "responsibilities", "duties", "task",
    "tasks", "job", "evaluation", "report", "education", "experience",
    "hire", "terminated", "leave", "vacation"
}


def extract_company_roles(user_question: str, chunks: list[str]) -> Optional[str]:
    """
    Extract exact executive role information (CEO / COO) ONLY when the user
    is specifically asking for the person's name or identity.

    Does NOT trigger for questions about salary, compensation, duties, contact details,
    or other attributes.
    """
    if not user_question or not chunks:
        return None

    question_lower = user_question.lower()

    # If the user is asking about salary, compensation, duties, etc., DO NOT use exact name lookup
    query_words = set(re.findall(r"\b\w+\b", question_lower))
    if any(non_name in query_words for non_name in NON_NAME_INTENTS):
        return None

    # Check for requested roles
    requested_roles: list[str] = []
    if re.search(r"\bceo\b", question_lower) or "chief executive officer" in question_lower:
        requested_roles.append("CEO")
    if re.search(r"\bcoo\b", question_lower) or "chief operating officer" in question_lower:
        requested_roles.append("COO")

    if not requested_roles:
        return None

    # Verify user is inquiring about identity/name/role
    # e.g., 'who is the ceo', 'name of ceo', 'who is coo', 'ceo name', 'who runs the company'
    identity_indicators = [
        "who", "name", "names", "person", "leader", "identity",
        "chair", "director", "head", "holds", "occupies"
    ]
    is_identity_query = (
        any(ind in query_words for ind in identity_indicators)
        or len(query_words) <= 3  # short query like 'ceo?' or 'the coo'
    )

    if not is_identity_query:
        return None

    found_roles: dict[str, str] = {}

    for chunk in chunks:
        for line in chunk.splitlines():
            clean_line = clean_text(line)
            if not clean_line:
                continue

            for role in requested_roles:
                if not re.search(rf"\b{role}\b", clean_line, re.IGNORECASE):
                    continue

                # Pattern matching: Name, Credentials - Role Description
                # e.g., 'Charu Raheja, PhD - Chair and CEO'
                #       'Ravi Raheja, MD - Medical Director and COO'
                name_match = re.search(
                    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)"
                    r",\s*(?:PhD|MD|MBA|JD|Esq\.?)"
                    r"\s*[-–—]\s*.*\b"
                    + role
                    + r"\b",
                    clean_line,
                    re.IGNORECASE
                )

                if name_match:
                    found_roles[role] = clean_line

    if not found_roles:
        return None

    answers: list[str] = []
    for role in requested_roles:
        if role in found_roles:
            answers.append(found_roles[role])
        else:
            answers.append(f"{role}: Information not found in the document.")

    return "\n".join(answers)
