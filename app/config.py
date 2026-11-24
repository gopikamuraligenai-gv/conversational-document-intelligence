from langchain_ollama import ChatOllama, OllamaEmbeddings


def initialize_llm():

    llm = ChatOllama(
        model="llama3.2",
        temperature=0.2
    )

    return llm


def initialize_embeddings():

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    return embeddings