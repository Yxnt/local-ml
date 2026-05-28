import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel

app = FastAPI(title="Jina Embedding Service")

MODEL_NAME = "jinaai/jina-embeddings-v5-omni-nano"

print(f"Loading {MODEL_NAME} ...")
model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    dtype=torch.float32,  # N100 无 GPU，用 float32
)
model.eval()
print("Model loaded.")


class EmbedRequest(BaseModel):
    texts: list[str]
    task: str = "retrieval"  # retrieval / text-matching / clustering


@app.post("/embed")
def embed(req: EmbedRequest):
    with torch.no_grad():
        embeddings = model.encode(
            req.texts,
            task=req.task,
            truncate_dim=768,  # 省内存，性能损失很小
        )
    return {"embeddings": embeddings.tolist()}


class EmbedBatchRequest(BaseModel):
    texts: list[str]
    task: str = "retrieval"


@app.post("/embed/batch")
def embed_batch(req: EmbedBatchRequest):
    """Batch embedding endpoint -- identical logic to /embed but explicit path."""
    with torch.no_grad():
        embeddings = model.encode(
            req.texts,
            task=req.task,
            truncate_dim=768,
        )
    return {"embeddings": embeddings.tolist()}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)