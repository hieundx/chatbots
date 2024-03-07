from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv
from langchain.tools.retriever import create_retriever_tool
from helpers import create_document_retriever, create_agent_executor

# Load environment variables from .env file
load_dotenv()

DOCUMENT_URLS = [
    'https://www.thebrokebackpacker.com/best-places-to-visit-in-ho-chi-minh/',
    'https://www.lonelyplanet.com/articles/best-things-to-do-in-ho-chi-minh-city',
    'https://vietnamtravel.com/top-ho-chi-minh-citys-attractions/'
]

# Load documents
retriever = create_document_retriever(DOCUMENT_URLS)

# Create tools
retriever_tool = create_retriever_tool(
    retriever,
    "hcmc-travels-retriever",
    "Retriever for HCMC travel documents",
)
tools = [retriever_tool]

# Create agent
agent_executor = create_agent_executor(tools)