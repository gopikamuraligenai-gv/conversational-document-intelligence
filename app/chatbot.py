from langchain_core.messages import HumanMessage, AIMessage


def ask_question(question, retriever, rag_chain, chat_history):

    # Retrieve top 2 relevant documents
    retrieved_docs = retriever.invoke(question)

    # Build context
    context = "\n\n".join(
        [
            f"Source: {doc.metadata.get('source', 'Unknown')}\n"
            f"Content: {doc.page_content}"
            for doc in retrieved_docs
        ]
    )

    # Convert chat history into text
    history_text = "\n".join(
        [
            f"{'Human' if isinstance(message, HumanMessage) else 'Assistant'}: "
            f"{message.content}"
            for message in chat_history
        ]
    )

    # Generate answer
    response = rag_chain.invoke(
        {
            "context": context,
            "question": question,
            "chat_history": history_text
        }
    )

    # Update conversation memory
    chat_history.append(
        HumanMessage(content=question)
    )

    chat_history.append(
        AIMessage(content=response)
    )

    return response


def start_chat(retriever, rag_chain):

    chat_history = []

    print("\n========================================")
    print("       CONVERSATIONAL RAG CHATBOT")
    print("========================================")

    print("\nType 'exit', 'quit', or 'stop' to finish.")

    while True:

        question = input("\nAsk your question: ").strip()

        # Check empty question
        if not question:
            print("Please enter a valid question.")
            continue

        # Check exit command
        if question.lower() in ["exit", "quit", "stop"]:

            print("\nFinal Conversation:")

            for message in chat_history:

                if isinstance(message, HumanMessage):
                    print(f"\nYou: {message.content}")

                elif isinstance(message, AIMessage):
                    print(f"Assistant: {message.content}")

            print("\nTotal messages in memory:", len(chat_history))
            print("\nChatbot stopped.")

            break

        try:

            answer = ask_question(
                question,
                retriever,
                rag_chain,
                chat_history
            )

            print("\nAnswer:")
            print(answer)

            print(
                "\nMemory Count:",
                len(chat_history)
            )

        except Exception as error:

            print("\nError:", error)
            print("Please try again.")