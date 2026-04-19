from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

# Permiso para que Next.js se conecte
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Aquí está la conexión a tu LM Studio local
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    senuelo_simulado = "jig metálico"

    # Personalidad y contexto para Qwen
    system_prompt = (
        "Eres un asistente virtual de inteligencia artificial, diseñado para proveer información técnica "
        "y estructurada sobre pesca deportiva. Responde de manera formal, concisa y objetiva. "
        f"Contexto del sistema: El usuario está consultando sobre un {senuelo_simulado}. "
        "Limítate a entregar datos útiles sobre su uso y especies objetivo."
    )

    try:
        # FastAPI le pide a LM Studio que genere la respuesta
        response = client.chat.completions.create(
            model="local-model",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.message}
            ],
            temperature=0.7,
        )

        return {
            "role": "assistant",
            "content": response.choices[0].message.content
        }
    except Exception as e:
        return {
            "role": "assistant",
            "content": f"Error conectando a LM Studio: {str(e)}"
        }
