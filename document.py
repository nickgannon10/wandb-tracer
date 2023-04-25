import pinecone
import openai
import os
from dotenv import load_dotenv
from tqdm.auto import tqdm
from time import sleep
from datasets import load_dataset

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


class DocumentIndexer:
    def __init__(self, dataset, index):
        self.dataset = dataset
        self.index = index
        self.batch_size = int(os.getenv("BATCH_SIZE"))
        self.embed_model = os.getenv("EMBED_MODEL")

    def _create_embeddings(self, texts):
        try:
            res = openai.Embedding.create(input=texts, engine=self.embed_model)
        except:
            done = False
            while not done:
                sleep(5)
                try:
                    res = openai.Embedding.create(input=texts, engine=self.embed_model)
                    done = True
                except:
                    pass
        return [record["embedding"] for record in res["data"]]

    def index_documents(self):
        documents = [
            {"id": doc["id"], "text": doc["text"], "url": doc["source"]}
            for doc in self.dataset
        ]

        for i in tqdm(range(0, len(documents), self.batch_size)):
            i_end = min(len(documents), i + self.batch_size)
            meta_batch = documents[i:i_end]
            ids_batch = [x["id"] for x in meta_batch]
            texts = [x["text"] for x in meta_batch]
            embeds = self._create_embeddings(texts)
            meta_batch = [{"text": x["text"], "url": x["url"]} for x in meta_batch]
            to_upsert = list(zip(ids_batch, embeds, meta_batch))
            self.index.upsert(vectors=to_upsert)


documents = load_dataset("Nickgannon10/langchain-docs", split="train")
indexer = DocumentIndexer(dataset=documents, index=index)
indexer.index_documents()
