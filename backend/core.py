from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Initialize embeddings (same as ingestion.py)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

#Initialize vector store
vectorstore = PineconeVectorStore(index_name="rag", embedding=embeddings)
# Initialize chat model
model = init_chat_model("gpt-5.2", model_provider="openai")


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer user queries about LangChain."""
    # Retrieve top 4 most similar documents
    retrieved_docs = vectorstore.as_retriever().invoke(query, k=4)
    
    # Serialize documents for the model
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    
    # Return both serialized content and raw documents
    return serialized, retrieved_docs


def run_llm(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation.
    
    Args:
        query: The user's question
        
    Returns:
        Dictionary containing:
            - answer: The generated answer
            - context: List of retrieved documents
    """
    # Create the agent with retrieval tool
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )
    
    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)
    
    # Build messages list
    messages = [{"role": "user", "content": query}]
    
    # Invoke the agent
    response = agent.invoke({"messages": messages})
    
    # Extract the answer from the last AI message
    answer = response["messages"][-1].content
    
    # Extract context documents from ToolMessage artifacts
    context_docs = []
    for message in response["messages"]:
        # Check if this is a ToolMessage with artifact
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            # The artifact should contain the list of Document objects
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)
    
    return {
        "answer": answer,
        "context": context_docs
    }

if __name__ == '__main__':
    result = run_llm(query="what are deep agents?")
    print(result)

"""
load_dotenv()	Loads environment variables, such as API keys, from the .env file.
OpenAIEmbeddings	Converts text into numerical vectors using the text-embedding-3-small embedding model.
PineconeVectorStore	Connects to the Pinecone vector database to search for previously indexed documentation.
index_name="rag"	Specifies the Pinecone index containing the embedded documents.
init_chat_model()	Initializes the LLM, here gpt-5.2, to generate answers and decide when to use tools.
@tool	Converts a Python function into a tool that the agent can call.
retrieve_context()	Searches the vector store for documentation relevant to the user's question.
response_format="content_and_artifact"	Returns both text content for the agent and a separate artifact containing the original retrieved documents.
as_retriever()	Provides a retriever interface for searching documents by semantic similarity.
invoke(query)	Executes the retrieval operation using the supplied query.
k=4	Intended to retrieve the four most relevant documents, but in this code k=4 is passed to invoke() rather than configured on the retriever, so it may not take effect.
doc.page_content	Contains the actual text extracted from a retrieved document.
doc.metadata	Contains additional information about a document, such as its source or file name.
serialized	Combines retrieved document text and source information into a string for the LLM.
content_and_artifact return	Returns (serialized, retrieved_docs), separating readable context from raw document objects.
create_agent()	Creates an agent that can use the LLM and the retrieve_context tool to answer questions.
system_prompt	Instructs the agent to retrieve documentation, cite sources, and admit when the answer is not found.
messages	Stores the conversation history supplied to the agent.
agent.invoke()	Executes the agent workflow, including any tool calls and subsequent LLM responses.
ToolMessage	Represents a message returned by a tool after the agent invokes it.
artifact	Stores the raw retrieved documents separately from the tool's text response.
context_docs	Collects retrieved Document objects so the caller can inspect the supporting context.
response["messages"][-1].content	Extracts the content of the final message, which is expected to contain the answer.
run_llm()	Runs the complete RAG agent workflow and returns the generated answer and retrieved context.
if __name__ == "__main__"	Runs the example query only when the Python file is executed directly.
"""