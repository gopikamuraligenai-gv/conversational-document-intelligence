from pathlib import Path
from langchain_core.documents import Document


def load_documents():

    # Get documents folder path
    project_root = Path(__file__).resolve().parent.parent

    documents_path = project_root / "documents"

    # Find all TXT files
    files = list(documents_path.glob("*.txt"))

    if not files:
        raise FileNotFoundError(
            "No TXT documents found in documents folder."
        )

    documents = []

    for file_path in files:

        print(f"Loading: {file_path.name}")

        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        if content.strip():
            document = Document(
                page_content=content,
                metadata={"source": file_path.name}
            )

            documents.append(document)

    print(f"\nTotal documents loaded: {len(documents)}")

    return documents