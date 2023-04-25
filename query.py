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

    def get_answer(self, query):
        query_with_contexts = self._retrieve(query)
        answer = self._complete(query_with_contexts)
        return answer


# load_dotenv()
# embed_model = os.getenv("EMBED_MODEL")
# pinecone_index = os.getenv("PINECONE_INDEX")


# index = pinecone.Index(pinecone_index)
# limit = 3750  # max number of characters in prompt
# query = "Tell me about ConversationBufferMemory in langchain, and walk me through how to use it."


# def retrieve(query):
#     res = openai.Embedding.create(input=[query], engine=embed_model)

#     # retrieve from Pinecone
#     xq = res["data"][0]["embedding"]

#     # get relevant contexts
#     res = index.query(xq, top_k=3, include_metadata=True)
#     contexts = [x["metadata"]["text"] for x in res["matches"]]

#     # build our prompt with the retrieved contexts included
#     prompt_start = "Answer the question based on the context below.\n\n" + "Context:\n"

#     prompt_end = f"\n\nQuestion: {query}\nAnswer:"
#     # append contexts until hitting limit
#     for i in range(1, len(contexts)):
#         if len("\n\n---\n\n".join(contexts[:i])) >= limit:
#             prompt = prompt_start + "\n\n---\n\n".join(contexts[: i - 1]) + prompt_end
#             break
#         elif i == len(contexts) - 1:
#             prompt = prompt_start + "\n\n---\n\n".join(contexts) + prompt_end
#     return prompt


# def complete(prompt):
#     # query text-davinci-003
#     res = openai.Completion.create(
#         engine="text-davinci-003",
#         prompt=prompt,
#         temperature=0,
#         max_tokens=400,
#         top_p=1,
#         frequency_penalty=0,
#         presence_penalty=0,
#         stop=None,
#     )
#     return res["choices"][0]["text"].strip()


# # first we retrieve relevant items from Pinecone
# query_with_contexts = retrieve(query)
# answer = complete(query_with_contexts)
# print(answer)
