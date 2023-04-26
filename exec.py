import pinecone
import os
from dotenv import load_dotenv
import openai
from datasets import load_dataset
from wandb.integration.langchain import WandbTracer
from langchain.llms import OpenAI
from query import PineconeQA
import tiktoken
import wandb

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_REGION")
model = os.getenv("MODEL")
embed_model = os.getenv("EMBED_MODEL")
pinecone_index = os.getenv("PINECONE_INDEX_NAME")
wandb_api = os.getenv("WANDB_API_KEY")


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens


pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
index = pinecone.Index(pinecone_index)
WandbTracer.init({"project": "wandb_prompts"})

qa = PineconeQA(index=index)

queries = [
    "Tell me about ConversationBufferMemory in langchain, and walk me through how to use it.",
    "What are chat bots and how do they work?",
    "Explain autonomous agents and their applications.",
    "Describe LLMChains and their role in langchain.",
    "What is Weights and Biases and how is it used in machine learning?",
]

# Initialize wandb run
run = wandb.init(project="pinecone_qa_logs")

# ✨ W&B: Create a Table to store query and answer
query_answer_table = wandb.Table(
    columns=[
        "inputs",
        "query",
        "answer",
        "prompt_tokens",
        "answer_tokens",
        "sum_tokens",
    ]
)

for query in queries:
    prompt = qa._retrieve(query=query)
    answer = qa._complete(prompt=prompt)
    prompt_tokens = num_tokens_from_string(prompt)
    answer_tokens = num_tokens_from_string(answer)
    sum_tokens = prompt_tokens + answer_tokens

    # ✨ W&B: Add query and answer to the table
    query_answer_table.add_data(
        prompt, query, answer, prompt_tokens, answer_tokens, sum_tokens
    )

# ✨ W&B: Log the table to wandb
wandb.log({"query_answer_table": query_answer_table})


# import pinecone
# import os
# from dotenv import load_dotenv
# import openai
# from datasets import load_dataset
# from wandb.integration.langchain import WandbTracer
# from langchain.llms import OpenAI
# from query import PineconeQA
# import tiktoken
# import wandb

# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")
# pinecone_api_key = os.getenv("PINECONE_API_KEY")
# pinecone_environment = os.getenv("PINECONE_REGION")
# model = os.getenv("MODEL")
# embed_model = os.getenv("EMBED_MODEL")
# pinecone_index = os.getenv("PINECONE_INDEX_NAME")
# wandb_api = os.getenv("WANDB_API_KEY")


# def num_tokens_from_string(string: str) -> int:
#     """Returns the number of tokens in a text string."""
#     encoding = tiktoken.get_encoding("cl100k_base")
#     num_tokens = len(encoding.encode(string))
#     return num_tokens


# pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
# index = pinecone.Index(pinecone_index)
# # run = wandb.init(project="pinecone_qa_logs")
# WandbTracer.init({"project": "wandb_prompts"})

# qa = PineconeQA(index=index)
# query = "Tell me about ConversationBufferMemory in langchain, and walk me through how to use it."
# prompt = qa._retrieve(query=query)
# answer = qa._complete(prompt=prompt)
# prompt_tokens = num_tokens_from_string(prompt)
# answer_tokens = num_tokens_from_string(answer)
# sum_tokens = prompt_tokens + answer_tokens


# # ✨ W&B: Create a Table to store query and answer
# query_answer_table = wandb.Table(
#     columns=[
#         "inputs",
#         "query",
#         "answer",
#         "prompt_tokens",
#         "answer_tokens",
#         "sum_tokens",
#     ]
# )

# # ✨ W&B: Add query and answer to the table
# query_answer_table.add_data(
#     prompt, query, answer, prompt_tokens, answer_tokens, sum_tokens
# )

# # ✨ W&B: Log the table to wandb
# wandb.log({"query_answer_table": query_answer_table})
