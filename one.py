import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

print("Tracing:", os.getenv("LANGSMITH_TRACING"))
print("LangSmith key loaded:", bool(os.getenv("LANGSMITH_API_KEY")))
print("Project:", os.getenv("LANGSMITH_PROJECT"))
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from tavily import TavilyClient
from langsmith import traceable

@traceable(name="langsmith_connection_test")
def test_tracing():
    return {"status": "LangSmith tracing test"}

print(test_tracing())

tavily=TavilyClient()

@tool
def search(query:str) ->str:
    """
    Tool that searches over the internet
    Args:
        query: The query to search for 
    Returns:
        The search result
    """
    print(f"searching for {query}")
    return tavily.search(query=query)
llm=ChatOpenAI(model="gpt-5")
tools=[search]
agent=create_agent(model=llm, tools=tools)

print("helo from langchain")
response = agent.invoke({"messages": HumanMessage(content="Find the top 3 latest AI Engineer jobs on LinkedIn in India. Include job title, company, location, experience required, and direct application link. Prioritize Python, GenAI, RAG, LangChain, Azure OpenAI, and AI agents. Prefer jobs posted within the last 7 days." )})


print(response["messages"][-1].content)