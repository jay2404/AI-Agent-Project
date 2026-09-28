from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

load_dotenv()

# Load existing vector store from disk (built in Day 4)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Retriever — fetches top-3 relevant chunks for any query
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# LLM
llm = ChatGroq(
model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
api_key=os.getenv("GROQ_API_KEY")
)

# Prompt — {context} = retrieved chunks, {question} = user query
prompt = ChatPromptTemplate.from_template("""
You are a helpful SQL expert assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "Not found in my knowledge base."

Context: {context}

Question: {question}
""")

# Helper to format retrieved docs into one string
def format_docs(docs):
 return "\n\n".join(d.page_content for d in docs)

# RAG Chain using LCEL pipe operator (you learned this in Week 1!)
rag_chain = (
{"context": retriever | format_docs, "question": RunnablePassthrough()}
| prompt
| llm
| StrOutputParser()
)

# Test it
questions = [
"What is the difference between RANK and DENSE_RANK?",
"How does LAG function work in SQL?",
"What is machine learning?" # not in our docs — should say "not found"
]
for q in questions:
 print(f"\nQ: {q}")
 print(f"A: {rag_chain.invoke(q)}")