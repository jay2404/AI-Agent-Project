from langchain_groq import ChatGroq
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()
llm =ChatGroq(model="openai/gpt-oss-20b", api_key=os.getenv("GROQ_API_KEY"))    

#Define the shape you want back
class SqlExplaination(BaseModel):
    sql_query: str
    explanation: str
    example: str
    difficulty: str


#Tell LLm to repsond in this structured format
structured_llm = llm.with_structured_output(SqlExplaination)

result = structured_llm.invoke("Explain SQL window functions")
print(f"SQL query: {result.sql_query}")
print(f"Explanation: {result.explanation}")
print(f"Example: {result.example}")