from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("data/sql_notes.txt")
docs = loader.load()

# RecursiveCharacterTextSplitter tries to split on paragraphs first,
# then sentences, then words — keeps context intact
splitter = RecursiveCharacterTextSplitter(
chunk_size=1000,
chunk_overlap=50,
add_start_index=True # tracks position in original doc
)

chunks = splitter.split_documents(docs)

print(f"Total chunks: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i+1} ---")
print(f"Length: {len(chunk.page_content)} chars")
print(chunk.page_content)
