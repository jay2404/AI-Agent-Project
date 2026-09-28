from langchain_huggingface import HuggingFaceEmbeddings

# Downloads model on first run (~90MB). After that it's cached.
embeddings = HuggingFaceEmbeddings(
model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Embed a single text — returns a list of 384 numbers
vector = embeddings.embed_query("What is ROW_NUMBER in SQL?")
print(f"Embedding dimensions: {len(vector)}")
print(f"First 5 values: {vector[:5]}")

# Embed multiple texts at once
texts = ["SQL window functions", "Python pandas", "SQL partitions"]
vectors = embeddings.embed_documents(texts)
print(f"Embedded {len(vectors)} texts")