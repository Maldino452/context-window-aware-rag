from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import glob


POLICIES_FOLDER = "policies"
DB_LOCATION = "./chroma_db"
EMBEDDING_MODEL = "mxbai-embed-large"


# WE FIND ALL .txt FILES IN THE POLICIES FOLDER
policy_files = glob.glob(os.path.join(POLICIES_FOLDER, "*.txt"))


# WE THEN BREAK LONG DOCUMENTS INTO SMALLER CHUNKS SO WE CAN RETRIEVE RELEVANT PARTS AND NOT ENTIRE DOCUMENTS
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)

all_documents = []
chunk_id = 0

for policy_file in policy_files:
    filename = os.path.basename(policy_file)
    
    with open(policy_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    chunks = text_splitter.split_text(content)
    for i, chunk_text in enumerate(chunks):
        document = Document(
            page_content=chunk_text,
            metadata={
                "source": filename,
                "chunk_id": f"{filename}_{i}"
            }
        )
        all_documents.append(document)
        chunk_id += 1


# WE THEN CREATE EMBEDDINGS AND VECTOR STORE
add_documents = not os.path.exists(DB_LOCATION)

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

vector_store = Chroma(
    collection_name="travel_expense_policies",
    persist_directory=DB_LOCATION,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=all_documents)


# WE THEN CREATE A RETRIEVER THAT WILL FIND AND RETURN THE TOP 6(k=6) MOST RELEVANT CHUNKS
retriever = vector_store.as_retriever(
    search_kwargs={"k": 6}
)

