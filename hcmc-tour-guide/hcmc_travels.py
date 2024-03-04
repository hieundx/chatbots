from typing import List

from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage
from langserve import add_routes
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

DOCUMENT_URLS = [
    'https://www.thebrokebackpacker.com/best-places-to-visit-in-ho-chi-minh/',
    'https://www.lonelyplanet.com/articles/best-things-to-do-in-ho-chi-minh-city',
    'https://vietnamtravel.com/top-ho-chi-minh-citys-attractions/'
]

from langchain.llms import OpenAI

llm = OpenAI()

loader = WebBaseLoader(DOCUMENT_URLS)

docs = loader.load()

embeddings = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()

from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "hcmc-travels-retriever",
    "Retriever for HCMC travel documents",
)
tools = [retriever_tool]

prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)