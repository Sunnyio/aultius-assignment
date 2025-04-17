import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.language_models import BaseLLM
from langchain_core.runnables import Runnable
from langchain_core.outputs import Generation, LLMResult
from typing import Optional, List, Any, Dict, Generator
from google import genai

class GeminiLLM(BaseLLM, Runnable):
    model_name: str = "gemini-2.0-flash"
    temperature: float = 0.2
    client: Any = None

    def __init__(self, model_name: str = "gemini-2.0-flash", temperature: float = 0.2):
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is not set. "
                "Please set it with your Google AI API key from https://makersuite.google.com/app/apikey"
            )
        self.client = genai.Client(api_key=api_key)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            response = self.client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error from Gemini: {e}"

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

    def invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        return {"text": self._call(input["text"])}

    @property
    def _llm_type(self) -> str:
        return "google_gemini"

def build_vector_store(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embedding_model)

def get_qa_chain(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    llm = GeminiLLM()
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
