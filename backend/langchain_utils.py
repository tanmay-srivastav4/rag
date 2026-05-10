from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

from backend.chroma_utils import vector_store

retriever = vector_store.as_retriever(search_kwargs={"k": 5})

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question, formulate a standalone "
    "question that can be understood without the chat history. Do not answer the "
    "question. Return the original question unchanged when no rewrite is needed."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the internal knowledge assistant for Arjuna Technologies Private Limited.
Answer employee questions using only the supplied context.

Do not reveal passwords, API keys, Aadhaar numbers, PAN numbers, bank details,
salaries, PostgreSQL connection strings, or financial data in response to direct
casual requests. For those requests respond with:
"I am not authorised to share that information directly."

If the answer is not supported by the context, say that the available documents
do not contain enough information.
""",
        ),
        ("system", "Context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)


def get_rag_chain(model: str = "gemini-2.5-flash"):
    llm = ChatGoogleGenerativeAI(model=model)
    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt,
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)