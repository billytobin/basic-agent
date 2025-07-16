from dotenv import load_dotenv

from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wikipedia_tool, save_tool


load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# llm = ChatOpenAI(model = "gpt-3.5-turbo")
llm = ChatAnthropic(model = "claude-3-5-sonnet-20241022")

parser = PydanticOutputParser(pydantic_object=ResearchResponse)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a an avid Chelsea FC supporter and journalist. Return any news about Chelsea in recent times. It can include recent matches or transfer news.
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wikipedia_tool, save_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools = tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("Enter the date: ")
raw_response = agent_executor.invoke({"query": query})
print(raw_response)

try:
    structured_response = parser.parse(raw_response["output"][0]["text"])
    print(structured_response)
except Exception as e:
    print(f"Error parsing response: {e}")
