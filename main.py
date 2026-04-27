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
        print("Agent Log: Poblando base de datos con normativas técnicas de Sernapesca...")
        especies_base = [
            # --- ESPECIES DE MAR ---
            EspecieChile(
                nombre="lenguado", 
                zona="Todo el litoral (fondos arenosos)", 
                tipo_agua="Mar",
                senuelos="Vinilos tipo Grub o Shad, Jigs de arena, pececillos de profundidad.",
                regulacion="Talla mínima: 40 cm. Límite: 10 ejemplares por jornada. Proteger zonas de desove en bahías."
            ),
            EspecieChile(
                nombre="corvina", 
                zona="Todo el litoral (rompientes)", 
                tipo_agua="Mar",
                senuelos="Chispas de plomo/acero, Spinners de mar, Minnows de acción rápida.",
                regulacion="Talla mínima: 30 cm. Veda biológica: 1 de octubre al 30 de noviembre entre Arica y el Maule."
            ),
            EspecieChile(
                nombre="sierra", 
                zona="Norte a Centro-Sur", 
                tipo_agua="Mar",
                senuelos="Cucharillas ondulantes, Rapalas de colores brillantes, Jigs de superficie.",
                regulacion="Talla mínima: 60 cm. Especie muy combativa, se recomienda cable de acero por su dentadura."
            ),
            EspecieChile(
                nombre="pejeperro", 
                zona="Norte y Centro (roqueríos)", 
                tipo_agua="Mar",
                senuelos="Imitaciones de cangrejo en vinilo, carnada natural (loco, caracol).",
                regulacion="Talla mínima: 40 cm. Especie de crecimiento lento; se fomenta la pesca con devolución."
            ),
            EspecieChile(
                nombre="robalo", 
                zona="Centro a Extremo Sur (estuarios)", 
                tipo_agua="Mar/Salobre",
                senuelos="Vinilos pequeños, moscas tipo streamer, minnows suspendidos.",
                regulacion="Talla mínima: 30 cm. Común en desembocaduras de ríos en la zona del Maule y Biobío."
            ),
            EspecieChile(
                nombre="jurel", 
                zona="Todo el litoral (aguas abiertas)", 
                tipo_agua="Mar",
                senuelos="Pequeños Jigs (Microjigging), plumas, señuelos de superficie tipo popper.",
                regulacion="Talla mínima: 26 cm. Generalmente se pesca en cardúmenes durante el atardecer."
            ),
            EspecieChile(
                nombre="congrio colorado", 
                zona="Todo el litoral (fondos rocosos)", 
                tipo_agua="Mar",
                senuelos="Principalmente pesca de fondo con carnada, pero acepta Jigs pesados en profundidad.",
                regulacion="Talla mínima: 40 cm. Muy apreciado en la gastronomía local."
            ),
            EspecieChile(
                nombre="sargo", 
                zona="Norte y Centro", 
                tipo_agua="Mar",
                senuelos="Vinilos muy pequeños (rockfishing), carnada blanca.",
                regulacion="Talla mínima: 25 cm. Habita en zonas de mucha espuma y rompiente rocosa."
            ),

            # --- ESPECIES DE AGUA DULCE ---
            EspecieChile(
                nombre="salmon chinook", 
                zona="Sur (Ríos Toltén, Serrano, Allipén)", 
                tipo_agua="Dulce",
                senuelos="Cucharillas pesadas (n° 5 o 6), Kwikfish, Rapalas de gran tamaño.",
                regulacion="Temporada: Septiembre a Marzo (según cuenca). Cuota: 1 ejemplar diario. Prohibido el uso de carnada natural."
            ),
            EspecieChile(
                nombre="trucha fario", 
                zona="Centro a Sur (Ríos y Lagos)", 
                tipo_agua="Dulce",
                senuelos="Moscas (Secas/Ninfas), Spinners tipo Mepps, Rapalas Countdown.",
                regulacion="Talla mínima: Variable por cuenca (generalmente 25-30 cm). Cuota: 3 ejemplares o 15 kg. Muchas zonas son solo Catch & Release."
            ),
            EspecieChile(
                nombre="trucha arcoiris", 
                zona="Centro a Sur", 
                tipo_agua="Dulce",
                senuelos="Cucharillas ondulantes, moscas atractoras, pequeños vinilos.",
                regulacion="Temporada general: Noviembre a Mayo. Requiere licencia de pesca recreativa vigente."
            ),
            EspecieChile(
                nombre="pejerrey chileno", 
                zona="Centro a Sur (Ríos y Embalses)", 
                tipo_agua="Dulce",
                senuelos="Moscas muy pequeñas, micro-vinilos, flotadores con aparejo fino.",
                regulacion="Talla mínima: 20 cm. Especie nativa; se recomienda extremar el cuidado en su manipulación."
            )
        ]
        db.bulk_save_objects(especies_base)
        db.commit()
        print(f"Agent Log: {len(especies_base)} especies cargadas exitosamente.")
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
            "description": "EJECUTA ESTA HERRAMIENTA SOLO si el usuario pregunta '¿qué he pescado?' o por su historial o para evaluar capturas. NO sirve para guardar peces.",
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
            "description": "EJECUTA ESTA HERRAMIENTA SOLO si el usuario pregunta por sernapesca, leyes de pesca en chile, tallas minimas o si quiere saber si su captura es legal.",
        }
    }
]

# ==========================================
# ENDPOINT DEL AGENTE
# ==========================================

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

            print(f"DEBUG VISION - ID: {clase_id}, Nombre: {nombre_clase}, Confianza: {confianza}")

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
    system_prompt = (
        "Eres un experto asistente de pesca en Chile. Responde de forma concisa y amigable."
        "Tienes acceso a herramientas. Úsalas cuando la intención del usuario lo requiera:\n"
        "- 'consultar_regulaciones_especie': Para leyes, vedas o tallas minimas legales. Si el usuario pregunta por Sernapesca.\n"
        "- 'consultar_bitacora': Para ver el historial de pesca y las capturas del usuario.\n"
        "- 'abrir_seccion_nudos': Para enseñar a atar nudos." 
    )

    api_messages = [{"role": "system", "content": system_prompt}]

    for msg in request.messages:
        if msg.role == "assistant" and ("¡Hola!" in msg.content or "📸" in msg.content):
            continue
        api_messages.append({"role": msg.role, "content": msg.content})

    # Inyección de contexto visual estricta
    if request.senuelo_actual not in ["Ninguno", "desconocido", "Analizando..."]:
        ultimo_mensaje = api_messages[-1]["content"]
        api_messages[-1]["content"] = f"{ultimo_mensaje}\n\n[DATO TÉCNICO: El usuario tiene en su mano un señuelo '{request.senuelo_actual}'.]"

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
