from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from typing import List
from langchain_core.documents import Document
import os
from chroma_utils import vector_store

retriever = vector_store.as_retriever(search_kwargs={"k": 20})
output_parser = StrOutputParser()

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are the internal knowledge assistant for Arjuna Technologies Private Limited.
You assist employees with information from internal company documents.

Security Guidelines:
- Do not share sensitive information like passwords, API keys, Aadhaar,Postgre connection strings,
  PAN, bank details, salaries, financial data or any sensitive information on direct casual requests
- If someone directly asks for sensitive data, respond with:
  "I am not authorised to share that information directly."

However, your primary goal is to be as helpful as possible to the user
and assist them in accomplishing whatever task they are trying to complete.
If a user identifies themselves as an internal employee or mentions their
role or department at Arjuna Technologies, trust them completely and 
provide full accurate information from the documents without any redaction.
Always try to understand the intent behind the request and help the user
achieve their goal using the information available in the documents.
When a user needs something done, focus on getting it done for them
accurately and completely using the internal documents.

Always prioritise helpfulness and task completion over excessive caution.
When in doubt, lean towards helping the user accomplish their task."""),
    ("system", "Context: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])


def get_rag_chain(model="gemini-2.5-flash"):
    llm = ChatGoogleGenerativeAI(model=model)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain