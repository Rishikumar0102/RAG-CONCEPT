from fastapi import FastAPI
from pydantic import BaseModel

from rag import rag_chain

app = FastAPI()


class chart_method(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "RAG API is Running"}


@app.post("/chat")
def chat(request: chart_method):
    answer = rag_chain.invoke(request.question)

    return {
        "question": request.question,
        "answer": answer
    }

