from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from agent.utils.llm import _get_embedding_model
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.tools.retriever import create_retriever_tool

def load_document(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

def split_text(pages):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.create_documents([page.page_content for page in pages])

def create_vectorstore(documents):
    return InMemoryVectorStore.from_documents(
        documents=documents,
        embedding=_get_embedding_model()
    )

def get_retriever_tool(vectorstore):
    retriever = vectorstore.as_retriever()
    return create_retriever_tool(
        retriever,
        "retrieve_hospital_information",
        "Search and return information about hospital and services.",
    )
