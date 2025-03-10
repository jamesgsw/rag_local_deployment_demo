# """
# RAG (Retrieval Augmented Generation) Application
#
# I reference this document to create this app: https://python.langchain.com/docs/tutorials/rag/#storing-documents
#
# This class implements a RAG system that:
# 1. Loads and processes documents
# 2. Creates embeddings
# 3. Stores them in a vector database
# 4. Retrieves relevant context
# 5. Generates responses using an LLM
#
# Authors: James Gan
# Date: December 2024
# """
#
# import os
# import yaml
# import logging
# from dotenv import load_dotenv
# from typing import TypedDict, List
#
# import bs4
# from langchain_chroma import Chroma
# from langchain_openai import ChatOpenAI
# from langgraph.graph import START, StateGraph
# from langchain_core.documents import Document
# from langchain_core.prompts import PromptTemplate
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
#
# with open("config.yaml", "r") as file:
#     config = yaml.safe_load(file)
#
# # Setting Log levels
# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )
#
# # Setting environment variables
# load_dotenv()
# # openai_api_key = os.getenv('OPENAI_API_KEY')
# # langsmith_api_key = os.getenv('LANGSMITH_API_KEY')
# # os.environ["USER_AGENT"] = "MyCustomUserAgent/1.0"
# os.environ["OPENAI_API_KEY"] = config["local_deployment_api"]["api_key"]
# os.environ["OPENAI_API_BASE"] = config["local_deployment_api"]["api_base"]
#
# llm = ChatOpenAI(
#     model="llama-3.3-70b-instruct"
# )
#
# os.environ["TOKENIZERS_PARALLELISM"]="false"
# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-mpnet-base-v2"
# )
#
# vector_store = Chroma(
#     embedding_function=embeddings
# )
#
#
# # Load and Chunk
# loader = WebBaseLoader(
#     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
#     bs_kwargs=dict(
#         parse_only=bs4.SoupStrainer(
#             class_=("post-content", "post-title", "post-header")
#         )
#     ),
# )
# docs = loader.load()
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# all_splits = text_splitter.split_documents(docs)
#
#
# # Index chunks
# _ = vector_store.add_documents(documents=all_splits)
#
#
#
# template = """In this task, you will be presented with {context} and {question}.
# Use the following pieces of {context} to answer the {question} at the end.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.
#
# Context: {context}
#
# Question: {question}
#
# Answer:
# """
# chat_prompt = PromptTemplate.from_template(template)
#
#
# ### Using non-LangSmith way
# question: str = "What is Task Decomposition?"
# retrieved_docs: List[Document] = vector_store.similarity_search(question)
# docs_content: str = "\n\n".join(doc.page_content for doc in retrieved_docs)
# prompt = chat_prompt.invoke({"question": question, "context": docs_content})
# answer = llm.invoke(prompt)
# print(answer.content)