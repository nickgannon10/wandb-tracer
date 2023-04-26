import pinecone
import os
from dotenv import load_dotenv
import openai
from datasets import load_dataset
import openai

load_dotenv()


class PineconeQA:
    def __init__(self, index):
        self.index = index
        self.embed_model = os.getenv("EMBED_MODEL")
        self.limit = int(os.getenv("LIMIT"))

    def _retrieve(self, query):
        res = openai.Embedding.create(input=[query], engine=self.embed_model)
        xq = res["data"][0]["embedding"]
        res = self.index.query(xq, top_k=3, include_metadata=True)
        contexts = [x["metadata"]["text"] for x in res["matches"]]

        prompt_start = (
            "Answer the question based on the context below.\n\n" + "Context:\n"
        )
        prompt_end = f"\n\nQuestion: {query}\nAnswer:"

        for i in range(1, len(contexts)):
            if len("\n\n---\n\n".join(contexts[:i])) >= self.limit:
                prompt = (
                    prompt_start + "\n\n---\n\n".join(contexts[: i - 1]) + prompt_end
                )
                break
            elif i == len(contexts) - 1:
                prompt = prompt_start + "\n\n---\n\n".join(contexts) + prompt_end
        return prompt

    def _complete(self, prompt):
        res = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=0,
            max_tokens=400,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
        )
        return res["choices"][0]["text"].strip()
