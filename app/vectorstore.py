from pathlib import Path
from langchain_community.vectorstores import FAISS


def create_vectorstore(documents, embeddings):

    print("\nGenerating embeddings...")

    # Create FAISS vector database
    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )

    print("Embeddings generated successfully!")

    # Save FAISS locally
    project_root = Path(__file__).resolve().parent.parent

    vectorstore_path = project_root / "vectorstore"

    vectorstore.save_local(str(vectorstore_path))

    print("FAISS Vector Store saved successfully!")

    return vectorstore