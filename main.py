from fastapi import FastAPI
from pydantic import BaseModel
import os
import openai

from rag.loader import load_scripture
from rag.retriever import build_index, retrieve_relevant_chunks
from rag.prompt import build_prompt

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI(
    title="Converse with Your God 🕉️",
    description="Scripture-grounded spiritual guidance using the Bhagavad Gita"
)

chunks = load_scripture()
index, chunks = build_index(chunks)

class QuestionRequest(BaseModel):
    question: str
    language: str  # "English" or "Hindi"


@app.get("/")
def health():
    return {"status": "running"}

@app.post("/ask")
def ask_god(req: QuestionRequest):
    relevant_chunks = retrieve_relevant_chunks(
        query=req.question,
        index=index,
        chunks=chunks,
        k=3
    )

    context = "\n\n".join(relevant_chunks)

    prompt = build_prompt(
        context=context,
        question=req.question,
        language=req.language
    )

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return {
        "answer": response.choices[0].message.content,
        "verses_used": relevant_chunks
    }
