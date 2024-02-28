from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import VLLM
import os

class LLMModel:
    def __init__(model_name):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if model_name != "gpt3.5":
            self.llm = VLLM(
                    model="DeepSeek-Coder-33B",
                    trust_remote_code=True,
                    max_new_tokens=1500,
                    top_k=9,
                    top_p=0.95,
                    temperature=0.1,
                    tensor_parallel_size=2  # for distributed inference
                )
        else:
            self.llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=openai_api_key)

    def complete(self, inputs, contexts):
        return self.llm(self.format(inputs, contexts))
    
    def format(self, inputs, contexts):
        return f"Given following context: {contexts} and your need to complete following {inputs} in one line:"