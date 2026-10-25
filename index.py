import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

pc.create_index(
    name="rag",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)
print("created")

"""
Pinecone – Creates a client connection to the Pinecone vector database using an API key.



ServerlessSpec – Configures the cloud provider and region for the serverless index.



create_index() – Creates a new Pinecone index to store and search vectors.



dimension=1536 – Sets the vector size to match the embedding model's output dimensions.



metric="cosine" – Uses cosine similarity to measure the similarity between vectors.



cloud="aws" – Selects AWS as the cloud provider hosting the index.



region="us-east-1" – Specifies the AWS region where the index is hosted.



load_dotenv() – Loads the Pinecone API key and other configuration values from .env.
"""