from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()
llm = ChatGroq(model="openai/gpt-oss-20b", api_key=os.getenv("GROQ_API_KEY"))

history = [SystemMessage(content="You are a International Relations expert. Be helpful and concise.")]

print("Chat started. Type 'quit' to exit.\n")
while True:
    user_input = input("You: ")
    if user_input == "quit":
        break
    history.append(HumanMessage(content=user_input))
    response = llm.invoke(history)
    history.append(AIMessage(content=response.content))
    print(f"AI: {response.content}\n")