from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama
import os

# Initialize FastAPI app
app = FastAPI()

try:
    llama = Llama.from_pretrained(
        repo_id="unsloth/Llama-3.2-1B-Instruct-GGUF",
        filename="Llama-3.2-1B-Instruct-F16.gguf",
    )
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

class Message(BaseModel):
    role: str  # "system", "user", or "assistant"
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: list[str] | None = None

class ChatResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class ChatResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[ChatResponseChoice]

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    """
    OpenAI-compatible endpoint for chat completions.
    """
    try:
        # Extract conversation history as prompt
        prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])

        # Generate response using LLaMA
        response = llama(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
        )
        generated_text = response["choices"][0]["text"]

        # Build response
        assistant_message = Message(role="assistant", content=generated_text)
        choice = ChatResponseChoice(index=0, message=assistant_message, finish_reason="stop")
        return ChatResponse(
            id="chatcmpl-unique-id",
            object="chat.completion",
            created=1234567890,
            model=request.model,
            choices=[choice],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "SillyTavern-Compatible LLM API Running!"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8001)))