from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# loader = TextLoader("data/sql_notes.txt")
# docs = loader.load()

loader = PyPDFLoader("data/T_C_Extended_Warranty_BS_VI-Up_to_5_years_w_o_RSA (2).pdf")
docs = loader.load() # returns one doc per page
# print(f"Loaded {len(docs)} document(s)")
print(f"Preview: {docs[0].page_content[:200]}")
print(f"Metadata: {docs[0].metadata}")

# --- If you have a PDF, load it like this ---


# --- Load ALL files in a folder ---
loader = DirectoryLoader("data/", glob="*.txt")
docs = loader.load()

print(f"Preview: {docs[0].page_content[:500]}")
print(f"Metadata: {docs[0].metadata}")