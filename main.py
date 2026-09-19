from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents import create_tool_calling_agent
from ai_agents.tools import search_tool, wiki_tool, save_tool

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

load_dotenv()
llm = ChatAnthropic(model = 'claude-sonnet-5')
parser = PydanticOutputParser(pydantic_object = ResearchResponse)

prompt = ChatPromptTemplate.from_messages(

[
  (
    "system",
    """
    You are a research assistant that will help generate a research paper. 
    Answer the user query and use the necessary tools. 
    Wrap the output in this format and provide no other textx \n{format_instructions}
    """,
),
("placeholder", "{chat_history}"),
("human", "{query}"),
("placeholder", "{agent_scratchpad}")

]

).partial(format_instructions = parser.get_format_instructions())
# response = llm.invoke('What is the meaning of life?')
# print(response)


tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools= tools, verbose =True)
query = input("What can l help you research today")
raw_response_agent= agent_executor.invoke({'query':query})
try: 
    structured_response = parser.parse(raw_response_agent.get("output"[0]["text"]))
    print(structured_response)
except Exception as e :
    print('Error parsing the response')
