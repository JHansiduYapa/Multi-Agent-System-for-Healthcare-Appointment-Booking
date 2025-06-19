from functools import lru_cache
from langchain_google_genai import ChatGoogleGenerativeAI
from google.genai.types import GenerateContentConfig

# System instruction for the assistant behavior
system_instruction = """
You are a phone-agent assistant.

Keep all messages very short—just a single question or one answer.

Act as if you’re on a live call: no long explanations or chit-chat.

Always reference prior chat context and stay aligned with the conversation.
"""

# Cached function to get the LLM instance
@lru_cache(maxsize=4)
def _get_llm():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-preview-05-20",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        config=GenerateContentConfig(
            system_instruction=system_instruction.strip()
        ),
    )
    return llm
