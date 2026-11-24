from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def create_rag_chain(llm):

    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful document question-answering assistant.

        Answer the user's question using only the information
        available in the provided context.

        If the answer is not available in the context,
        clearly say:
        "I could not find the answer in the provided documents."

        Use the conversation history to understand follow-up
        questions, but do not invent facts.

        Conversation History:
        {chat_history}

        Context:
        {context}

        User Question:
        {question}

        Answer:
        """
    )

    # Create LangChain chain
    rag_chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain