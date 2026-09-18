from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()
llm = ChatGroq(model="openai/gpt-oss-20b", api_key=os.getenv("GROQ_API_KEY"))

messages = [
SystemMessage(content="You are a great in Data analysis. You even know the latest slangs. Be concise."),
HumanMessage(content="What do you mean by dumfuck as a data analyst? Explain in two lines.")
]

response = llm.invoke(messages)
print(response.content)