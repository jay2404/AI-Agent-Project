from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
llm = ChatGroq(model="openai/gpt-oss-20b", api_key=os.getenv("GROQ_API_KEY"))

# Define a reusable prompt template
prompt = ChatPromptTemplate.from_messages([
("system", "You are a SQL expert. Explain concepts clearly."),
("human", "Explain {topic} with a SQL example.")
])

# Chain: prompt | llm | output parser (pipe operator)
chain = prompt | llm | StrOutputParser()

# Run it — pass variables to the template
result = chain.invoke({"topic": "window functions"})
print(result)

# Try different topics
result2 = chain.invoke({"topic": "CTEs vs subqueries"})
print(result2)