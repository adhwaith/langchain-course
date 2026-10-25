import os

from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")
    loader = UnstructuredLoader(file_path="C:/Users/Lenovo/Desktop/rag_langchain/mediumblog1.txt", chunking_strategy="basic", max_characters=1000000)
    document = loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    print("ingesting...")
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ["INDEX_NAME"]
    )
    print("finish")


"""
load_dotenv() – Loads environment variables such as API keys from the .env file.



UnstructuredLoader – Loads and extracts text content from the source file.



CharacterTextSplitter – Splits the document into smaller chunks of approximately 1,000 characters.



chunk_overlap=0 – Prevents overlapping text between consecutive chunks.



split_documents() – Splits the loaded documents while preserving their document structure and metadata.



OpenAIEmbeddings – Converts each text chunk into a numerical vector representing its semantic meaning.



PineconeVectorStore.from_documents() – Generates embeddings for the chunks and stores them in the Pinecone index.



INDEX_NAME – Specifies which Pinecone index receives the document vectors.
"""