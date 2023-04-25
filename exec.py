import pinecone
import os
from dotenv import load_dotenv
import openai
from datasets import load_dataset

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_REGION")
model = os.getenv("MODEL")
embed_model = os.getenv("EMBED_MODEL")
pinecone_index = os.getenv("PINECONE_INDEX_NAME")

pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
index = pinecone.Index(pinecone_index)

query = "Tell me about ConversationBufferMemory in langchain, and walk me through how to use it."

from query import PineconeQA

qa = PineconeQA(index=index)
query = "Tell me about ConversationBufferMemory in langchain, and walk me through how to use it."
answer = qa.get_answer(query)
print(answer)
