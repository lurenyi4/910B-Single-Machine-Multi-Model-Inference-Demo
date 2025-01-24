from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from llamafactory.chat import ChatModel
import argparse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def create_chat_model(model_args: Optional[Dict[str, Any]] = None):
    return ChatModel(model_args)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, 
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--infer_backend", type=str, default="huggingface", 
                        choices=["huggingface", "vllm"], help="Inference backend to use")
    parser.add_argument("--template", type=str, required=True,
                        help="Template to use for the model (e.g. chatglm3, llama2, vicuna)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    return parser.parse_args()

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    system: Optional[str] = None
    tools: Optional[str] = None
    image: Optional[str] = None
    video: Optional[str] = None
    max_new_tokens: Optional[int] = 512
    top_p: Optional[float] = 0.7
    temperature: Optional[float] = 0.95

class ChatResponse(BaseModel):
    response: str
    finish_reason: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        responses = await chat_model.achat(
            messages=request.messages,
            system=request.system,
            tools=request.tools,
            image=request.image,
            video=request.video,
            max_new_tokens=request.max_new_tokens,
            top_p=request.top_p,
            temperature=request.temperature
        )
        if responses:
            return ChatResponse(
                response=responses[0].response_text,
                finish_reason=responses[0].finish_reason
            )
        else:
            raise HTTPException(status_code=500, detail="No response from model")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stream_chat")
async def stream_chat(request: ChatRequest):
    try:
        async def generate():
            async for new_token in chat_model.astream_chat(
                messages=request.messages,
                system=request.system,
                tools=request.tools,
                image=request.image,
                video=request.video,
                max_new_tokens=request.max_new_tokens,
                top_p=request.top_p,
                temperature=request.temperature
            ):
                yield new_token

        return generate()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    args = parse_args()
    model_args = {
        "model_name_or_path": args.model_name_or_path,
        "infer_backend": args.infer_backend,
        "template": args.template
    }
    chat_model = create_chat_model(model_args)
    uvicorn.run(app, host=args.host, port=args.port) 