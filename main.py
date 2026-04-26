from fastapi import FastAPI, File, UploadFile, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from database import SessionLocal, Captura
from pydantic import BaseModel
from openai import OpenAI
from typing import List
from ultralytics import YOLO
from PIL import Image
import io
import shutil
import os
import datetime

# pylint: disable=no-member
import numpy as np

app = FastAPI()

if not os.path.exists("uploads"):
    os.makedirs("uploads")

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


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
        "Eres un asistente virtual experto en pesca deportiva. Tu objetivo es ser útil y mantener "
        "una conversación fluida con el usuario. Responde sus preguntas directamente. "
        f"Contexto del sistema: El usuario tiene actualmente un {ultimo_senuelo_detectado}. "
        "Usa esta información para dar contexto cuando hables de pesca, pero eres libre de responder otras preguntas de forma natural."
    )

    api_messages = [{"role": "system", "content": system_prompt}]

    for msg in request.messages:
        api_messages.append({"role": msg.role, "content": msg.content})

    try:
        response = client.chat.completions.create(
            model="local-model",
            messages=api_messages,
            temperature=0.7,
        )
        return {"role": "assistant", "content": response.choices[0].message.content}
    except Exception as e:
        return {"role": "assistant", "content": f"Error conectando a LM Studio: {str(e)}"}


@app.post("/guardar-captura")
async def guardar_captura(
    especie: str = Form(...),
    medida: float = Form(...),
    senuelo: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    try:
        # 1. Generar un nombre único para la imagen
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"{timestamp}_{file.filename}"
        ruta_final = os.path.join("uploads", nombre_archivo)

        # 2. Guardar el archivo físicamente
        with open(ruta_final, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 3. Guardar el registro en la base de datos
        nueva_captura = Captura(
            especie=especie,
            medida_cm=medida,
            senuelo=senuelo,
            ruta_imagen=nombre_archivo
        )
        db.add(nueva_captura)
        db.commit()
        db.refresh(nueva_captura)

        return {"status": "success", "message": "Captura guardada en la bitácora", "id": nueva_captura.id}

    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/historial")
async def obtener_historial(db: Session = Depends(get_db)):
    # Traemos todas las capturas ordenadas por la más reciente
    capturas = db.query(Captura).order_by(Captura.fecha.desc()).all()
    return capturas
