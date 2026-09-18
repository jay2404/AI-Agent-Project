from fastapi import FastAPI
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()
app = FastAPI(title="My AI Chatbot")
llm = ChatGroq(model="openai/gpt-oss-20b", api_key=os.getenv("GROQ_API_KEY"))

class ChatRequest(BaseModel):
	message: str
	history: list[dict] = []

class ChatResponse(BaseModel):
	reply: str

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
	messages = [SystemMessage(content="You are a helpful data analyst.")]
	for h in req.history:
		if h["role"] == "user":
			messages.append(HumanMessage(content=h["content"]))
		else:
			messages.append(AIMessage(content=h["content"]))
	messages.append(HumanMessage(content=req.message))
	response = llm.invoke(messages)
	return ChatResponse(reply=response.content)

@app.get("/")
def health(): return {"status": "running"}