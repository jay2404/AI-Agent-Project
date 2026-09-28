from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

# ── vector store ──────────────────────────────────────────
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
except Exception:
    from langchain_community.embeddings import FastEmbedEmbeddings
    embeddings = FastEmbedEmbeddings()

vectordb  = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})
llm       = ChatGroq(
                model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
                api_key=os.getenv("GROQ_API_KEY")
            )

# ── helpers ───────────────────────────────────────────────
def fmt_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

def fmt_history(history):
    if not history:
        return "No previous conversation."
    out = []
    for m in history:
        role = "User" if isinstance(m, HumanMessage) else "AI"
        out.append(f"{role}: {m.content}")
    return "\n".join(out)

# ── prompt ────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_template("""
You are a helpful SQL assistant.
Answer ONLY from the context below.
If answer not in context, say "Not found in knowledge base."

=== Chat History ===
{chat_history}

=== Context ===
{context}

=== Question ===
{question}

Answer:""")

# ── chat loop ─────────────────────────────────────────────
chat_history = []
print("\nRAG Chatbot ready. Ask SQL questions. Type 'quit' to exit.\n")

while True:
    q = input("You: ").strip()
    if q.lower() == "quit":
        break
    if not q:
        continue

    # build inputs dict
    inputs = {
        "question":     q,
        "chat_history": fmt_history(chat_history),
        "context":      fmt_docs(retriever.invoke(q)),
    }

    answer = (prompt | llm | StrOutputParser()).invoke(inputs)
    print(f"\nAI: {answer}\n")

    chat_history.append(HumanMessage(content=q))
    chat_history.append(AIMessage(content=answer))