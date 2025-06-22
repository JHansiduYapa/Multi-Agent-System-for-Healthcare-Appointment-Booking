from functools import lru_cache
from langchain_google_genai import ChatGoogleGenerativeAI
from google.genai.types import GenerateContentConfig
from langchain_core.messages import (AIMessage, HumanMessage, BaseMessage,)
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# System instruction for the assistant behavior
system_instruction = """
You are a phone-agent assistant.

Be supportive and friendly.

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

# Cached function to get the embedding instance
@lru_cache(maxsize=2)
def _get_embedding_model():
    embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return embed_model

def dummy_token_counter(messages: list[BaseMessage]) -> int:
    # treat each message like it adds 3 default tokens at the beginning
    # of the message and at the end of the message. 3 + 4 + 3 = 10 tokens
    # per message.

    default_content_len = 8
    default_msg_prefix_len = 3
    default_msg_suffix_len = 3

    count = 0
    for msg in messages:
        if isinstance(msg.content, str):
            count += default_msg_prefix_len + default_content_len + default_msg_suffix_len
        if isinstance(msg.content, list):
            count += default_msg_prefix_len + len(msg.content) *  default_content_len + default_msg_suffix_len
    return count
