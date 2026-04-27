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
import re
import json

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

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

print("Cargando modelo de visión...")
try:
    modelo_vision = YOLO("best.pt")
    print("¡Modelo cargado exitosamente!")
except Exception as e:
    print(f"Error al cargar best.pt: {e}")

ultimo_senuelo_detectado = "desconocido"


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    senuelo_actual: str = "desconocido"

# ==========================================
# HERRAMIENTAS DEL AGENTE (FUNCTION CALLING)
# ==========================================


def consultar_bitacora_db():
    print("Agent Log: El LLM solicitó acceso a la base de datos...")
    db = SessionLocal()
    try:
        capturas = db.query(Captura).order_by(
            Captura.fecha.desc()).limit(10).all()

        if not capturas:
            return "El historial está vacío. Aún no hay peces registrados en la bitácora."

        resultado = "Historial de capturas recientes del usuario:\n"
        for c in capturas:
            fecha_str = c.fecha.strftime(
                '%d-%m-%Y') if c.fecha else 'Fecha desconocida'
            resultado += f"- Especie: {c.especie}, Medida: {c.medida_cm}cm, Señuelo: {c.senuelo}, Fecha: {fecha_str}\n"

        return resultado
    except Exception as e:
        return f"Error interno al consultar la base de datos: {str(e)}"
    finally:
        db.close()


herramientas_agente = [
    {
        "type": "function",
        "function": {
            "name": "consultar_bitacora",
            "description": "Busca en la base de datos local el historial de peces capturados. Úsala si el usuario pregunta por sus capturas.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }
    },
    {
        "type": "function",
        "function": {
            "name": "abrir_seccion_nudos",
            "description": "Redirige al usuario a la sección de videos tutoriales de nudos. Úsala si el usuario pregunta cómo hacer un nudo o atar el sedal.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }
    }
]


@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    global ultimo_senuelo_detectado

    try:
        image_data = await file.read()
        imagen = Image.open(io.BytesIO(image_data))
        resultados = modelo_vision(imagen)

        if len(resultados) > 0 and len(resultados[0].boxes) > 0:
            mejor_caja = resultados[0].boxes[0]
            clase_id = int(mejor_caja.cls[0].item())
            confianza = float(mejor_caja.conf[0].item())
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


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # Un system prompt más corto y directo ayuda a los modelos pequeños
    system_prompt = (
        "Eres un asistente de pesca en Chile. Responde brevemente en Markdown. "
        "Especies locales: Corvina, Lenguado, Trucha, Salmón. "
        "Herramientas: 'consultar_bitacora' (historial) y 'abrir_seccion_nudos' (nudos)."
    )

    api_messages = [{"role": "system", "content": system_prompt}]

    for msg in request.messages:
        if msg.role == "assistant" and ("¡Hola!" in msg.content or "📸" in msg.content):
            continue
        api_messages.append({"role": msg.role, "content": msg.content})

    # --- EL HACK PARA MODELOS PEQUEÑOS ---
    # En lugar de poner el señuelo arriba, se lo anexamos de forma invisible
    # al último mensaje del usuario para obligarlo a leerlo.
    if request.senuelo_actual not in ["Ninguno", "desconocido", "Analizando..."]:
        ultimo_mensaje = api_messages[-1]["content"]
        api_messages[-1]["content"] = f"{ultimo_mensaje}\n\n[Nota del sistema: El usuario tiene un señuelo tipo {request.senuelo_actual}. Basa tu respuesta en esto.]"

    try:
        response = client.chat.completions.create(
            model="local-model",
            messages=api_messages,
            temperature=0.7,
            tools=herramientas_agente,
            tool_choice="auto"
        )

        response_message = response.choices[0].message

        # Variable para capturar la acción de redirección
        action_flag = None

        if response_message.tool_calls:
            api_messages.append(response_message)

            for tool_call in response_message.tool_calls:
                # Caso 1: Consulta a la DB
                if tool_call.function.name == "consultar_bitacora":
                    resultado_db = consultar_bitacora_db()
                    api_messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": "consultar_bitacora",
                        "content": resultado_db,
                    })

                # Caso 2: Redirección a Nudos
                elif tool_call.function.name == "abrir_seccion_nudos":
                    action_flag = "navigate_nudos"
                    api_messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": "abrir_seccion_nudos",
                        "content": "OK. Dile al usuario que lo estás llevando a la sección de nudos.",
                    })

            # Generamos la respuesta final considerando los resultados de las herramientas
            final_response = client.chat.completions.create(
                model="local-model",
                messages=api_messages,
                temperature=0.7,
            )
            raw_content = final_response.choices[0].message.content
        else:
            raw_content = response_message.content
            if "abrir_seccion_nudos" in raw_content or "seccion_nudos" in raw_content:
                print("Agent Log: Tool Leak detectado. Forzando redirección manual.")
                action_flag = "navigate_nudos"
                # Limpiamos la respuesta para que el usuario no lea el texto feo
                raw_content = raw_content.replace(
                    'abrir_seccion_nudos', 'nuestra Academia de Nudos')
                raw_content = raw_content.replace(
                    'seccion_nudos', 'nuestra Academia de Nudos')

            if "consultar_bitacora" in raw_content:
                # Si filtró la bitácora, consultamos manualmente y añadimos el texto
                datos_extra = consultar_bitacora_db()
                raw_content = raw_content.replace(
                    'consultar_bitacora', 'tu bitácora') + f"\n\n*Nota del sistema: {datos_extra}*"

        clean_content = re.sub(r'<think>.*?</think>\n*',
                               '', raw_content, flags=re.DOTALL).strip()

        # Devolvemos la respuesta con el campo 'action' para el frontend
        return {
            "role": "assistant",
            "content": clean_content,
            "action": action_flag
        }

    except Exception as e:
        return {"role": "assistant", "content": f"Error: {str(e)}"}


@app.post("/guardar-captura")
async def guardar_captura(
    especie: str = Form(...),
    medida: float = Form(...),
    senuelo: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"{timestamp}_{file.filename}"
        ruta_final = os.path.join("uploads", nombre_archivo)

        with open(ruta_final, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        nueva_captura = Captura(
            especie=especie, medida_cm=medida, senuelo=senuelo, ruta_imagen=nombre_archivo)
        db.add(nueva_captura)
        db.commit()
        db.refresh(nueva_captura)
        return {"status": "success", "message": "Captura guardada", "id": nueva_captura.id}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/historial")
async def obtener_historial(db: Session = Depends(get_db)):
    return db.query(Captura).order_by(Captura.fecha.desc()).all()


@app.delete("/captura/{captura_id}")
async def borrar_captura(captura_id: int, db: Session = Depends(get_db)):
    try:
        # 1. Buscar la captura en la base de datos
        captura = db.query(Captura).filter(Captura.id == captura_id).first()
        if not captura:
            return {"status": "error", "message": "Captura no encontrada"}

        # 2. Borrar el archivo de imagen físico si existe
        ruta_archivo = os.path.join("uploads", captura.ruta_imagen)
        if os.path.exists(ruta_archivo):
            os.remove(ruta_archivo)

        # 3. Borrar el registro de la base de datos
        db.delete(captura)
        db.commit()

        return {"status": "success", "message": "Captura eliminada correctamente"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
