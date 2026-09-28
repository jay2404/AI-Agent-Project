from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Step 1: Load
loader = TextLoader("data/sql_notes.txt")
docs = loader.load()

# Step 2: Split
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"Created {len(chunks)} chunks")

# Step 3: Embed
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 4: Store in Chroma (persists to disk in ./chroma_db folder)
vectordb = Chroma.from_documents(
documents=chunks,
embedding=embeddings,
persist_directory="./chroma_db"
)
print(f"Stored {vectordb._collection.count()} chunks in Chroma")

# Test similarity search — this is what RAG uses
query = "How does ROW_NUMBER work?"
results = vectordb.similarity_search(query, k=2)
print(f"\nTop 2 results for: '{query}'")
for r in results:
 print(f"\n{r.page_content}")

 