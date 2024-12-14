from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama
from transformers import BartForConditionalGeneration, BartTokenizer

import os

# Initialize FastAPI app
app = FastAPI()

try:
    llama = Llama.from_pretrained(
        repo_id="unsloth/Llama-3.2-1B-Instruct-GGUF",
        filename="Llama-3.2-1B-Instruct-F16.gguf",
    )
    bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
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

def summarize_with_bart(text: str):
    inputs = bart_tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = bart_model.generate(inputs, max_length=50, min_length=10, length_penalty=2.0, num_beams=4)
    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    """
    OpenAI-compatible endpoint for chat completions.
    """
    try:
        # Extract conversation history as prompt
        prompt = "system: You are a helpful assistant.\n"
        for msg in request.messages:
            prompt += f"{msg.role}: {msg.content}\n"
        prompt += "assistant:"

        # Generate response from the model
        response = llama(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=["user:", "assistant:"],  # Stop generating when roles appear
        )
        print(f"Raw model response: {response}")

        # Extract the assistant's message
        generated_text = summarize_with_bart(response["choices"][0]["text"].strip())

        # Construct the assistant message
        assistant_message = Message(role="assistant", content=generated_text)
        print(f"Assistant message: {assistant_message}")
        # Return the response
        choice = ChatResponseChoice(index=0, message=assistant_message, finish_reason="stop")
        return ChatResponse(
            id=response["id"],
            object="chat.completion",
            created=response["created"],
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