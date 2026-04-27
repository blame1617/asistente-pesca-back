from fastapi import FastAPI, File, UploadFile, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from database import SessionLocal, Captura, EspecieChile
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
# FUNCIONES DE BASE DE DATOS
# ==========================================
def consultar_bitacora_db():
    print("Agent Log: Accediendo a bitácora...")
    db = SessionLocal()
    try:
        capturas = db.query(Captura).order_by(
            Captura.fecha.desc()).limit(10).all()
        if not capturas:
            return "El historial está vacío."
        resultado = "Historial de capturas:\n"
        for c in capturas:
            fecha_str = c.fecha.strftime('%d-%m-%Y') if c.fecha else 'S/D'
            resultado += f"- {c.especie}, {c.medida_cm}cm, Señuelo: {c.senuelo} ({fecha_str})\n"
        return resultado
    except Exception as e:
        return f"Error en DB: {str(e)}"
    finally:
        db.close()


def consultar_regulaciones_db():
    print("Agent Log: Accediendo a Sernapesca...")
    db = SessionLocal()
    try:
        especies = db.query(EspecieChile).all()
        resultado = "Leyes de Sernapesca:\n"
        for e in especies:
            resultado += f"- {e.nombre.capitalize()}: {e.regulacion}\n"
        return resultado
    finally:
        db.close()


def inicializar_conocimiento_pesca():
    db = SessionLocal()
    if db.query(EspecieChile).count() == 0:
        especies_base = [
            EspecieChile(nombre="lenguado", zona="Todo el litoral", tipo_agua="Mar",
                         senuelos="Vinilos, Jigs", regulacion="Talla mínima: 40 cm. Límite: 10 ejemplares."),
            EspecieChile(nombre="corvina", zona="Todo el litoral", tipo_agua="Mar",
                         senuelos="Chispas, Spinners", regulacion="Veda: 1 oct al 30 nov (Arica a Magallanes)."),
            EspecieChile(nombre="salmon chinook", zona="Sur", tipo_agua="Dulce",
                         senuelos="Cucharillas, Rapalas", regulacion="Temporada 15 sept a 31 marzo. 1 ejemplar diario."),
            EspecieChile(nombre="trucha", zona="Sur y Cordillera", tipo_agua="Dulce",
                         senuelos="Moscas, Spinners", regulacion="Cuota: 3 ejemplares o 15 kilos. Veda en invierno.")
        ]
        db.bulk_save_objects(especies_base)
        db.commit()
    db.close()


inicializar_conocimiento_pesca()

# ==========================================
# HERRAMIENTAS (BLINDADAS)
# ==========================================
# Las descripciones ahora son instrucciones hiper-estrictas
herramientas_agente = [
    {
        "type": "function",
        "function": {
            "name": "consultar_bitacora",
            "description": "EJECUTA ESTA HERRAMIENTA SOLO si el usuario pregunta '¿qué he pescado?' o por su historial. NO sirve para guardar peces.",
        }
    },
    {
        "type": "function",
        "function": {
            "name": "abrir_seccion_nudos",
            "description": "EJECUTA ESTA HERRAMIENTA SOLO si el usuario quiere aprender a atar un nudo. NO la uses para guardar capturas ni para otra cosa.",
        }
    },
    {
        "type": "function",
        "function": {
            "name": "consultar_regulaciones_especie",
            "description": "EJECUTA ESTA HERRAMIENTA SOLO si el usuario pregunta por leyes, vedas o tallas mínimas permitidas.",
        }
    }
]

# ==========================================
# ENDPOINT DEL AGENTE
# ==========================================


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    system_prompt = (
        "Eres un asistente de pesca en Chile. Responde brevemente en Markdown.\n"
        "REGLAS:\n"
        "1. NO uses herramientas para guardar peces.\n"
        "2. Si preguntan qué señuelo tienen, mira el contexto visual y responde sin usar herramientas."
    )

    api_messages = [{"role": "system", "content": system_prompt}]

    for msg in request.messages:
        if msg.role == "assistant" and ("¡Hola!" in msg.content or "📸" in msg.content):
            continue
        api_messages.append({"role": msg.role, "content": msg.content})

    # Inyección de contexto visual estricta
    if request.senuelo_actual not in ["Ninguno", "desconocido", "Analizando..."]:
        ultimo_mensaje = api_messages[-1]["content"]
        api_messages[-1]["content"] = f"{ultimo_mensaje}\n\n[CONTEXTO VISUAL: El usuario muestra un {request.senuelo_actual}.]"

    try:
        response = client.chat.completions.create(
            model="local-model",
            messages=api_messages,
            temperature=0.7,
            tools=herramientas_agente,
            tool_choice="auto"
        )

        response_message = response.choices[0].message
        action_flag = None
        raw_content = ""

        # CASO 1: EL MODELO USA LA HERRAMIENTA CORRECTAMENTE (JSON)
        if response_message.tool_calls:
            api_messages.append(response_message)

            for tool_call in response_message.tool_calls:
                func_name = tool_call.function.name
                print(
                    f"Agent Log: Herramienta invocada correctamente -> {func_name}")

                if func_name == "consultar_bitacora":
                    resultado = consultar_bitacora_db()
                elif func_name == "consultar_regulaciones_especie":
                    resultado = consultar_regulaciones_db()
                elif func_name == "abrir_seccion_nudos":
                    resultado = "Redirigiendo a nudos..."
                    action_flag = "navigate_nudos"
                else:
                    resultado = "Herramienta inválida."

                api_messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": func_name,
                    "content": resultado,
                })

            final_response = client.chat.completions.create(
                model="local-model",
                messages=api_messages,
                temperature=0.7,
            )
            raw_content = final_response.choices[0].message.content

        # CASO 2: EL MODELO FALLA Y ESCUPE TEXTO O XML (INTERCEPTOR HÍBRIDO)
        else:
            raw_content = response_message.content

            # Si el modelo imprime las etiquetas XML de la herramienta en el chat
            if "<tool_call>" in raw_content or "abrir_seccion_nudos" in raw_content or "consultar_" in raw_content:
                print("Agent Log: Fuga de Tool Call detectada. Interceptando...")

                if "abrir_seccion_nudos" in raw_content:
                    action_flag = "navigate_nudos"
                    raw_content = re.sub(
                        r'<tool_call>.*?</tool_call>', '¡Enseguida! Te redirijo a la Academia de Nudos.', raw_content, flags=re.DOTALL)
                    raw_content = raw_content.replace(
                        "abrir_seccion_nudos", "la sección de nudos")

                elif "consultar_regulaciones_especie" in raw_content:
                    datos = consultar_regulaciones_db()
                    raw_content = re.sub(
                        r'<tool_call>.*?</tool_call>', f'Consultando la base de datos...\n\n{datos}', raw_content, flags=re.DOTALL)
                    raw_content = raw_content.replace(
                        "consultar_regulaciones_especie", "las regulaciones oficiales")

                elif "consultar_bitacora" in raw_content:
                    datos = consultar_bitacora_db()
                    raw_content = re.sub(
                        r'<tool_call>.*?</tool_call>', f'Revisando tu historial...\n\n{datos}', raw_content, flags=re.DOTALL)
                    raw_content = raw_content.replace(
                        "consultar_bitacora", "tu bitácora")

        clean_content = re.sub(r'<think>.*?</think>\n*',
                               '', raw_content, flags=re.DOTALL).strip()

        return {
            "role": "assistant",
            "content": clean_content,
            "action": action_flag
        }

    except Exception as e:
        return {"role": "assistant", "content": f"Error conectando al LLM: {str(e)}"}


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
        captura = db.query(Captura).filter(Captura.id == captura_id).first()
        if not captura:
            return {"status": "error", "message": "Captura no encontrada"}

        ruta_archivo = os.path.join("uploads", captura.ruta_imagen)
        if os.path.exists(ruta_archivo):
            os.remove(ruta_archivo)

        db.delete(captura)
        db.commit()
        return {"status": "success", "message": "Captura eliminada correctamente"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
