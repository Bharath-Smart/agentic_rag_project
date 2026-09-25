
from langchain.agents import create_agent
from .prompts import SYSTEM_PROMPT
from .providers import get_llm_model
from .tools import (
    vector_search_tool,
    hybrid_search_tool,
    get_document_tool,
    list_documents_tool,
)




rag_agent = create_agent(
    model=get_llm_model(),
    tools=[
        vector_search_tool,
        hybrid_search_tool,
        get_document_tool,
        list_documents_tool,
    ],
    system_prompt=SYSTEM_PROMPT,
)

