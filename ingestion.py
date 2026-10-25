import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import (Colors, log_error, log_header, log_info, log_success,
                    log_warning)

load_dotenv()

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=False,
    chunk_size=50,
    retry_min_seconds=10,
)
#vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
# ingestion.py and the retrieval file
vectorstore = PineconeVectorStore(index_name="rag", embedding=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """Process documents in batches asynchronously."""
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"📚 VectorStore Indexing: Preparing to add {len(documents)} documents to vector store",
        Colors.DARKCYAN,
    )

    # Create batches
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"📦 VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )

    # Process all batches concurrently
    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vectorstore.aadd_documents(batch)
            log_success(
                f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
            )
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True

    # Process batches concurrently
        # Process batches one at a time (Pinecone's async session isn't safe to share)
    successful = 0
    for i, batch in enumerate(batches):
        if await add_batch(batch, i + 1):
            successful += 1

    if successful == len(batches):
        log_success(
            f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully"
        )


async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "🗺️  TavilyCrawl: Starting to crawl the documentation site",
        Colors.PURPLE,
    )
    # Crawl the documentation site

    res = tavily_crawl.invoke(
        {
            "url": "https://python.langchain.com/",
            "max_depth": 2,
            "extract_depth": "advanced",
        }
    )

    # Convert Tavily crawl results to LangChain Document objects
    all_docs = []

    for item in res.get("results", []):
        content = item.get("raw_content")
        url = item.get("url", "unknown")

        if not isinstance(content, str) or not content.strip():
            log_warning(f"Skipping page with no content: {url}")
            continue

        all_docs.append(
            Document(
                page_content=content,
                metadata={"source": url},
            )
        )

    log_success(f"Created {len(all_docs)} documents from crawl results")

    if not all_docs:
        log_error("No documents with valid content were extracted.")
        return

    # Split documents into chunks
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"✂️  Text Splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap",
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )

    # Process documents asynchronously
    await index_documents_async(splitted_docs, batch_size=100)

    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(splitted_docs)}")


if __name__ == "__main__":
    asyncio.run(main())


"""
Purpose: Crawls the LangChain docs site, splits the pages into chunks, and stores them in a vector database for semantic search.
load_dotenv(): Loads API keys (OpenAI, Tavily) from the .env file into the environment.
certifi / SSL settings: Points Python at certifi’s trusted certificates so HTTPS requests don’t fail with SSL errors.
OpenAIEmbeddings: Converts text into vectors using text-embedding-3-small so meaning can be compared numerically.
chunk_size=50 (embeddings): Controls how many texts are sent to OpenAI per embedding request.
retry_min_seconds=10: Sets the minimum wait before retrying a failed embedding request.
Chroma: A local vector store that saves embeddings in the chroma_db folder.--changed to pinecone itself
PineconeVectorStore (commented out): The cloud alternative to Chroma, currently unused.
TavilyCrawl: Visits the docs site and returns the text of each page it finds.
TavilyMap / TavilyExtract: Created but not used; they map a site’s structure and extract page content respectively.
Document: LangChain’s container that holds a page’s text (page_content) plus its metadata such as the source URL.
metadata={"source": url}: Records which page each piece of text came from.
RecursiveCharacterTextSplitter: Cuts long pages into smaller chunks, splitting at natural boundaries like paragraphs first.
chunk_size=4000: Sets the maximum size of each chunk to about 4000 characters.
chunk_overlap=200: Repeats 200 characters between neighboring chunks so ideas aren’t cut off at the edges.
Batches: Groups of chunks (500 each) that are indexed together to avoid API limits and isolate failures.
aadd_documents: Asynchronously embeds a batch of chunks and saves them into the vector store.
asyncio.gather: Runs all batch uploads concurrently instead of one after another.
return_exceptions=True: Prevents one failed batch from crashing the others.
index_documents_async: Splits chunks into batches, uploads them, and logs how many batches succeeded.
main(): Orchestrates the full pipeline: crawl, convert to documents, split, index, and print a summary.
logger functions: Print formatted progress messages (header, info, success, warning, error) from the local logger.py file.
Requirements: Needs OPENAI_API_KEY and TAVILY_API_KEY in .env, plus the langchain-classic package for the splitter import.

in main streamlit code is there : python.exe -m streamlit run main.py
"""