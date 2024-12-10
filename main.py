from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama
import os

# Initialize FastAPI app
app = FastAPI()

# Load the model (adjust the path to your GGUF model)
MODEL_PATH = "model/model.gguf"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

try:
    llama = Llama(model_path=MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Request and response models
class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.9

class InferenceResponse(BaseModel):
    generated_text: str

@app.post("/inference", response_model=InferenceResponse)
async def generate_text(request: InferenceRequest):
    """
    Endpoint for model inference.
    """
    try:
        response = llama(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
        )
        generated_text = response["choices"][0]["text"]
        return InferenceResponse(generated_text=generated_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Welcome to the LLM Inference API!"}
