from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv() # loads your .env file

# Create the LLM
llm = ChatGroq(
model="openai/gpt-oss-120b",
# openai/gpt-oss-120b
api_key=os.getenv("GROQ_API_KEY")
)

# Ask it a question
response = llm.invoke("What is a vector database? Explain in 2 lines.")
response1= llm.invoke("How to teach my friend English vocabulary?My friend is a 26 year old dumbfuck.Explain in 5 lines.")
# print(response.content)
print(response1.content)