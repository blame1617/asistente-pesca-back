from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from typing import List
from ultralytics import YOLO
from PIL import Image
import io
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cliente para tu LLM Local (Qwen / Llama)
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# --- CARGA DEL MODELO DE VISIÓN LOCAL ---
# Esto reemplaza al "senuelo_simulado" hardcodeado
print("Cargando modelo de visión...")
try:
    modelo_vision = YOLO("best.pt")
    print("¡Modelo cargado exitosamente!")
except Exception as e:
    print(f"Error al cargar best.pt: {e}")

# Variable global temporal para guardar el último señuelo detectado
ultimo_senuelo_detectado = "desconocido"


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]

# 1. NUEVO ENDPOINT DE VISIÓN (Totalmente Offline)


@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    global ultimo_senuelo_detectado

    try:
        # Leemos la imagen que envía el frontend
        image_data = await file.read()
        imagen = Image.open(io.BytesIO(image_data))

        # Pasamos la imagen a la IA de visión
        resultados = modelo_vision(imagen)

        # Extraemos los datos (Buscamos la caja con mayor confianza)
        if len(resultados) > 0 and len(resultados[0].boxes) > 0:
            mejor_caja = resultados[0].boxes[0]
            clase_id = int(mejor_caja.cls[0].item())
            confianza = float(mejor_caja.conf[0].item())

            # Convertimos el ID a nombre (ej: 0 -> "Cuchara")
            nombre_clase = resultados[0].names[clase_id]

            ultimo_senuelo_detectado = nombre_clase

            return {
                "status": "success",
                "detected": ultimo_senuelo_detectado,
                "confidence": round(confianza, 2)
            }
        else:
            return {"status": "not_found", "message": "No detecté ningún señuelo."}

    except Exception as e:
        return {"status": "error", "message": str(e)}

# 2. ENDPOINT DE CHAT (Actualizado)


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    global ultimo_senuelo_detectado

    # Ahora el LLM usa el señuelo REAL que detectó la cámara
    system_prompt = (
            "Eres un asistente virtual experto en pesca deportiva estrictamente en Chile. "
            "Usa formato Markdown para estructurar tus respuestas. "
            "REGLA DE ORO: Solo puedes recomendar las siguientes especies locales según la zona: "
            "- Zona Centro/Norte (Mar): Corvina, Lenguado, Sierra, Róbalo, Pejerrey de mar. "
            "- Zona Sur/Lagos/Ríos (Agua dulce): Trucha Fario, Trucha Arcoíris, Salmón Chinook, Salmón Coho. "
            f"Contexto de visión: El usuario te está mostrando un señuelo tipo: {ultimo_senuelo_detectado}. "
            "Relaciona este señuelo ESPECÍFICAMENTE con las especies chilenas listadas arriba y el tipo de fondo (arena, roquerío, río). "
            "No inventes especies que no existan en esta lista.")

    api_messages = [{"role": "system", "content": system_prompt}]

    for msg in request.messages:

        if msg.role == "assistant" and ("¡Hola! Soy tu asistente" in msg.content or "📸 ¡Listo!" in msg.content):
            continue
        api_messages.append({"role": msg.role, "content": msg.content})

    try:
        response = client.chat.completions.create(
            model="local-model",
            messages=api_messages,
            temperature=0.7,
        )

        raw_content = response.choices[0].message.content
        clean_content = re.sub(r'<think>.*?</think>\n*', '', raw_content, flags=re.DOTALL).strip()
        return {"role": "assistant", "content": clean_content}
    except Exception as e:
        return {"role": "assistant", "content": f"Error conectando a LM Studio: {str(e)}"}
