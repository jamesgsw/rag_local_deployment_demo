"""
RAG (Retrieval Augmented Generation) Application

I reference this document to create this app: https://python.langchain.com/docs/tutorials/rag/#storing-documents

This class implements a RAG system that:
1. Loads and processes documents
2. Creates embeddings
3. Stores them in a vector database
4. Retrieves relevant context
5. Generates responses using an LLM

Authors: James Gan
Date: December 2024
"""

import os
import yaml
import logging
from dotenv import load_dotenv
from typing import List

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

with open("src/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Setting Log levels
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)


# Setting environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')


# Initialise the main tools in LLM RAG Application
llm = ChatOpenAI(
    model=config["llm"]["llm_model"]["model_name"],
    temperature=config["llm"]["llm_model"]["temperature"]
)
embeddings = OpenAIEmbeddings(
    model=config["llm"]["embedding_model"]["model_name"]
)
vector_store = Chroma(
    embedding_function=embeddings
)


# Load and Chunk
loader: PyPDFLoader = PyPDFLoader(config["documents"]["file_path"])
documents: List[Document] = loader.load() # Load the PDF after reading in
text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
) # Initialise text splitter
split_docs: List[Document] = text_splitter.split_documents(documents) # Split documents into chunks


# Index chunks
_ = vector_store.add_documents(documents=split_docs)


template = """In this task, you will be presented with {context} and {question}.
Use the following pieces of {context} to answer the {question} at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer:
"""
chat_prompt = PromptTemplate.from_template(template)

question: str = input("Type in your question into the LLM? ")   # What is Task Decomposition?
retrieved_docs: List[Document] = vector_store.similarity_search(question) # Retrieve documents from vector database
docs_content: str = "\n\n".join(doc.page_content for doc in retrieved_docs)
prompt = chat_prompt.invoke({"question": question, "context": docs_content})
answer = llm.invoke(prompt)
print(answer.content)