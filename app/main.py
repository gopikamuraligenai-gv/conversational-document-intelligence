
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

from app.config import initialize_llm, initialize_embeddings
from app.loader import load_documents
from app.vectorstore import create_vectorstore
from app.rag_chain import create_rag_chain
from app.chatbot import start_chat


def main():

    print("=" * 50)
    print(" CONVERSATIONAL DOCUMENT INTELLIGENCE")
    print("=" * 50)

    # Load environment variables
    load_dotenv(PROJECT_ROOT / ".env")

    print("\nInitializing Ollama LLM...")
    llm = initialize_llm()

    print("Initializing Ollama Embeddings...")
    embeddings = initialize_embeddings()

    print("Loading Documents...")
    documents = load_documents()

    print(f"Loaded {len(documents)} documents")

    print("\nCreating FAISS Vector Store...")
    vectorstore = create_vectorstore(documents, embeddings)

    print("Creating Retriever...")
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 2}
    )

    print("Creating RAG Chain...")
    rag_chain = create_rag_chain(llm)

    print("\n" + "=" * 50)
    print(" RAG APPLICATION READY!")
    print("=" * 50)

    start_chat(retriever, rag_chain)


if __name__ == "__main__":
    main()
