# import os
# from typing import TypedDict, List
#
# from langgraph.graph import START, StateGraph
# from langchain_core.documents import Document
#
#
# ### Using LangSmith and LangGraph to build RAG Application
# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = langsmith_api_key
# class State(TypedDict):
#     question: str
#     context: List[Document]
#     answer: str
#
# # Define application steps
# def retrieve(state: State):
#     retrieved_docs = vector_store.similarity_search(state["question"])
#     return {"context": retrieved_docs}
#
# def generate(state: State):
#     docs_content: str = "\n\n".join(doc.page_content for doc in state["context"])
#     messages = chat_prompt.invoke({"question": state["question"], "context": docs_content})
#     response = llm.invoke(messages)
#     return {"answer": response.content}
#
# graph_builder = StateGraph(State).add_sequence([retrieve, generate])
# graph_builder.add_edge(START, "retrieve")
# graph = graph_builder.compile()
#
# response = graph.invoke({"question": "What is Task Decomposition?"})
# print(response["answer"])