from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from database import SessionLocal, Captura, EspecieChile, PuntoPescaChile
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
from dotenv import load_dotenv
from openai import AzureOpenAI
import requests
import time
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents import AgentsClient
import psycopg2
from decimal import Decimal

# pylint: disable=no-member
import numpy as np

# Cargar variables de entorno
load_dotenv()

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

print("Configurando ecosistema híbrido de IA...")

# --- CLIENTE 1: Orquestador (Rápido, Stateless, Modo OpenAI v1 original) ---
try:
    endpoint_base = os.getenv("AZURE_OPENAI_ENDPOINT", "")

    # Truco maestro: Asegurarnos de que el endpoint termine sí o sí en /openai/v1
    # para no tener que usar api_version jamás.
    if not endpoint_base.endswith("/openai/v1"):
        endpoint_base = f"{endpoint_base.rstrip('/')}/openai/v1"

    orquestador_client = OpenAI(
        base_url=endpoint_base,
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    print("✅ Orquestador configurado exitosamente (Modo OpenAI v1).")
except Exception as e:
    print(f"❌ Error en Orquestador: {e}")
    orquestador_client = None

# --- CLIENTE 2: Sub-Agentes (Nuevo estándar Foundry Responses API) ---
try:
    project_client = AIProjectClient(
        endpoint=os.getenv("AZURE_AI_PROJECT_ENDPOINT"),
        credential=DefaultAzureCredential()
    )
    # Extraemos el cliente mágico que maneja la Responses API
    agentes_client = project_client.get_openai_client()
    print("✅ AI Projects Client configurado exitosamente.")
except Exception as e:
    print(f"❌ Error en AI Projects Client: {e}")
    agentes_client = None

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

# Definimos el modelo de datos que esperamos recibir del Orquestador


class ParametrosClima(BaseModel):
    ubicacion: str
    solicitud_especifica: str = ""

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
        return "Error al acceder a la bitácora."

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


def consultar_puntos_pesca_db():
    print("Agent Log: Accediendo al Atlas GIS Offline...")
    db = SessionLocal()
    try:
        puntos = db.query(PuntoPescaChile).all()
        resultado = "Puntos y Hotspots de Pesca en Chile:\n"
        for p in puntos:
            resultado += f"- **{p.nombre}** ({p.region}): Tipo de agua: {p.tipo_agua}. Especies: {p.especies_objetivo}. Táctica: {p.recomendacion_tecnica} (Lat: {p.latitud}, Lon: {p.longitud})\n"
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
        print(
            f"Agent Log: {len(especies_base)} especies cargadas exitosamente.")
    db.close()


def inicializar_puntos_pesca():
    db = SessionLocal()
    if db.query(PuntoPescaChile).count() == 0:
        print("Agent Log: Poblando base de datos con 32 Hotspots GIS de pesca en Chile (2 por región)...")
        puntos_base = [
            # 1. Región de Arica y Parinacota
            PuntoPescaChile(nombre="Desembocadura Río Lluta (Playa Las Machas)", region="Región de Arica y Parinacota", latitud=-18.416, longitud=-70.322, tipo_agua="Mar/Salobre",
                            especies_objetivo="Corvina, Lenguado", recomendacion_tecnica="Ideal para surfcasting de orilla. Usar chispas metálicas y minnows lanzados detrás de la primera rompiente."),
            PuntoPescaChile(nombre="Playa Corazones / Cuevas de Anzota", region="Región de Arica y Parinacota", latitud=-18.525, longitud=-70.330, tipo_agua="Mar",
                            especies_objetivo="Pejeperro, Cabrilla, Sargo", recomendacion_tecnica="Roqueríos profundos. Pesca a fondo con plomo fusible o rockfishing con vinilos imitando pequeños crustáceos."),

            # 2. Región de Tarapacá
            PuntoPescaChile(nombre="Punta Gruesa", region="Región de Tarapacá", latitud=-20.360, longitud=-70.160, tipo_agua="Mar",
                            especies_objetivo="Corvina, Lenguado, Cabrilla", recomendacion_tecnica="Zonas mixtas de arena y roca. Utilizar señuelos sinking rápidos y jigs pesados para evitar enredos en los sargazos."),
            PuntoPescaChile(nombre="Playa Chanavayita", region="Región de Tarapacá", latitud=-20.700, longitud=-70.190, tipo_agua="Mar",
                            especies_objetivo="Lenguado de gran porte, Sargo", recomendacion_tecnica="Buscar pozones y canaletas cercanas a las rocas de los extremos usando vinilos con cabeza plomada de 20g a 30g."),

            # 3. Región de Antofagasta
            PuntoPescaChile(nombre="Punta Angamos (Mejillones)", region="Región de Antofagasta", latitud=-23.050, longitud=-70.470, tipo_agua="Mar",
                            especies_objetivo="Dorado (verano), Sierra, Jurel", recomendacion_tecnica="Aguas profundas ideales para spinning pesado con rapalas de babero largo o casting de orilla con jigs metálicos."),
            PuntoPescaChile(nombre="Sector Hornitos / Punta Hornos", region="Región de Antofagasta", latitud=-22.900, longitud=-70.280, tipo_agua="Mar",
                            especies_objetivo="Corvina, Lenguado", recomendacion_tecnica="Extensas playas de pendiente fuerte. Trabajar chispas ondulantes y paseantes hundidos en los remolinos de corriente."),

            # 4. Región de Atacama
            PuntoPescaChile(nombre="Bahía Inglesa / Las Machas", region="Región de Atacama", latitud=-27.080, longitud=-70.860, tipo_agua="Mar",
                            especies_objetivo="Lenguado gigante, Corvina", recomendacion_tecnica="Rastrear los fondos de arena blanca con vinilos tipo Shad o Grub color chartreuse/blanco de 4 a 5 pulgadas."),
            PuntoPescaChile(nombre="Caleta Pan de Azúcar (Roqueríos permitidos)", region="Región de Atacama", latitud=-26.140, longitud=-70.660, tipo_agua="Mar",
                            especies_objetivo="Pejeperro, Sierra, Cabrilla", recomendacion_tecnica="Roqueríos de alta productividad para spinning con jigs y vinilos plomados. Respetar áreas exclusivas del Parque Nacional."),

            # 5. Región de Coquimbo
            PuntoPescaChile(nombre="Punta Lengua de Vaca (Tongoy)", region="Región de Coquimbo", latitud=-30.250, longitud=-71.580, tipo_agua="Mar",
                            especies_objetivo="Sierra trofeo, Corvina, Jurel", recomendacion_tecnica="Fuertes corrientes oceánicas. Indispensable uso de cable de acero para sierras y chispas de acción errática."),
            PuntoPescaChile(nombre="Desembocadura Río Limarí", region="Región de Coquimbo", latitud=-30.730, longitud=-71.680, tipo_agua="Mar/Salobre",
                            especies_objetivo="Lenguado, Pejerrey de mar", recomendacion_tecnica="Pescar en la barra de arena durante la estoa de marea baja con señuelos de hundimiento lento y recogidas pausadas."),

            # 6. Región de Valparaíso
            PuntoPescaChile(nombre="Desembocadura Río Maipo (Santo Domingo)", region="Región de Valparaíso", latitud=-33.611, longitud=-71.625, tipo_agua="Mar/Salobre",
                            especies_objetivo="Lenguado, Corvina, Robalo", recomendacion_tecnica="Puntos de encuentro de agua dulce y salada. Excelente para minnows de 12-15 cm y vinilos pesados en pozones de salida."),
            PuntoPescaChile(nombre="Playa Las Docas / Laguna Verde", region="Región de Valparaíso", latitud=-33.110, longitud=-71.670, tipo_agua="Mar",
                            especies_objetivo="Pejeperro, Sargo, Corvina", recomendacion_tecnica="Roqueríos escarpados de difícil acceso. Extremar precauciones con el oleaje. Usar chispas de 50g+ para alcanzar distancia."),

            # 7. Región Metropolitana de Santiago
            PuntoPescaChile(nombre="Embalse El Yeso (Cajón del Maipo)", region="Región Metropolitana", latitud=-33.670, longitud=-70.090, tipo_agua="Dulce",
                            especies_objetivo="Trucha Fario, Trucha Arcoíris", recomendacion_tecnica="Pesca de altura (2500m msnm). Usar cucharillas pequeñas (Mepps nº1 o 2) o moscas secas al atardecer. Agua extremadamente fría."),
            PuntoPescaChile(nombre="Río Maipo (Tramo San José de Maipo)", region="Región Metropolitana", latitud=-33.640, longitud=-70.350, tipo_agua="Dulce",
                            especies_objetivo="Trucha Arcoíris, Pejerrey nativo", recomendacion_tecnica="Pozones rocosos del tramo alto. Lanzar ninfas con indicador de picada o pequeños vinilos plomados en corrientes contrarias."),

            # 8. Región de O'Higgins
            PuntoPescaChile(nombre="Pichilemu (Puntilla / Infiernillo)", region="Región de O'Higgins", latitud=-34.390, longitud=-72.010, tipo_agua="Mar",
                            especies_objetivo="Corvina, Robalo, Sargo", recomendacion_tecnica="Fuerte oleaje y rompiente surfera. Spinning de orilla con señuelos pesados que soporten la turbulencia sin desestabilizarse."),
            PuntoPescaChile(nombre="Lago Rapel (Sector Las Balsas)", region="Región de O'Higgins", latitud=-34.130, longitud=-71.460, tipo_agua="Dulce",
                            especies_objetivo="Pejerrey argentino, Carpa", recomendacion_tecnica="Pesca embarcada o desde muelles. Uso de líneas muy finas, flotadores sensibles y moscas diminutas o carnada autorizada."),

            # 9. Región del Maule
            PuntoPescaChile(nombre="Sector Curanipe / Cardonal", region="Región del Maule", latitud=-35.845, longitud=-72.585, tipo_agua="Mar",
                            especies_objetivo="Corvina, Lenguado, Robalo", recomendacion_tecnica="Considerada una de las capitales del surfcasting. Buscar canaletones paralelos a la playa lanzando chispas de 40-60g."),
            PuntoPescaChile(nombre="Desembocadura Río Maule (Constitución)", region="Región del Maule", latitud=-35.330, longitud=-72.420, tipo_agua="Mar/Salobre",
                            especies_objetivo="Robalo gigante, Lenguado, Lisa", recomendacion_tecnica="Grandes remolinos en la barra. Usar bucktails, jigs pesados y minnows de profundidad en los cambios de marea."),

            # 10. Región de Ñuble
            PuntoPescaChile(nombre="Cobquecura (Piedra de la Iglesia)", region="Región de Ñuble", latitud=-36.130, longitud=-72.800, tipo_agua="Mar",
                            especies_objetivo="Corvina, Robalo, Sargo", recomendacion_tecnica="Playas de arena negra y fuerte resaca. Lanzar detrás de la barra de arena con señuelos compactos y aerodinámicos."),
            PuntoPescaChile(nombre="Río Diguillín (Sector Recinto)", region="Región de Ñuble", latitud=-36.880, longitud=-71.650, tipo_agua="Dulce",
                            especies_objetivo="Trucha Fario nativa, Trucha Arcoíris", recomendacion_tecnica="Tramos de aguas cristalinas y pozones rocosos. Pesca fina con equipo mosquero ligero (#3/#4) o spinning ultralight."),

            # 11. Región del Biobío
            PuntoPescaChile(nombre="Desembocadura Río Biobío (Hualpén)", region="Región del Biobío", latitud=-36.815, longitud=-73.165, tipo_agua="Mar/Salobre",
                            especies_objetivo="Robalo trofeo, Lenguado, Corvina", recomendacion_tecnica="Gran estuario. Rastrear los verilones de arena con vinilos montados en anzuelos offset para evitar enganches."),
            PuntoPescaChile(nombre="Península de Tumbes / Cantera", region="Región del Biobío", latitud=-36.620, longitud=-73.080, tipo_agua="Mar",
                            especies_objetivo="Sierra, Jurel, Sargo", recomendacion_tecnica="Acantilados profundos. Uso de chispas plateadas recogidas a gran velocidad para detonar la agresividad de los pelágicos."),

            # 12. Región de La Araucanía
            PuntoPescaChile(nombre="Desembocadura Río Toltén (La Barra)", region="Región de La Araucanía", latitud=-39.248, longitud=-73.220, tipo_agua="Dulce/Salobre",
                            especies_objetivo="Salmón Chinook, Robalo, Trucha", recomendacion_tecnica="Epicentro del Chinook en Chile. Requiere cañas heavy (2.7m+), trenzado de 50lb+ y cucharillas pesadas nº 5 o 6."),
            PuntoPescaChile(nombre="Río Allipén (Melipeuco)", region="Región de La Araucanía", latitud=-38.920, longitud=-72.030, tipo_agua="Dulce",
                            especies_objetivo="Salmón Chinook, Trucha Fario", recomendacion_tecnica="Zona de desove en curso alto. Derivar señuelos tipo Kwikfish (K14/K15) o rapalas articuladas cerca del fondo de los pozones."),

            # 13. Región de Los Ríos
            PuntoPescaChile(nombre="Río San Pedro (Desagüe Lago Riñihue)", region="Región de Los Ríos", latitud=-39.770, longitud=-72.450, tipo_agua="Dulce",
                            especies_objetivo="Trucha Arcoíris, Trucha Fario, Chinook", recomendacion_tecnica="Famoso por grandes truchas residentes. Flotadas en balsa lanzando streamers articulados o rapalas hacia troncos sumergidos."),
            PuntoPescaChile(nombre="Desembocadura Río Bueno", region="Región de Los Ríos", latitud=-40.240, longitud=-73.710, tipo_agua="Mar/Salobre",
                            especies_objetivo="Salmón Chinook trofeo, Robalo", recomendacion_tecnica="Amplio caudal de salida al mar. Castear desde orilla o bote con jigs pesados y cañas de acción rápida para clavar firme."),

            # 14. Región de Los Lagos
            PuntoPescaChile(nombre="Lago Llanquihue (Desembocadura Río Pescado)", region="Región de Los Lagos", latitud=-41.280, longitud=-72.850, tipo_agua="Dulce",
                            especies_objetivo="Trucha Fario, Trucha Arcoíris, Salmón Coho", recomendacion_tecnica="Zonas de veril abrupto. Ideal al amanecer y atardecer con moscas atractoras o cucharas ondulantes finas."),
            PuntoPescaChile(nombre="Río Petrohué (Sector Saltos)", region="Región de Los Lagos", latitud=-41.160, longitud=-72.400, tipo_agua="Dulce",
                            especies_objetivo="Salmón Chinook, Trucha Arcoíris", recomendacion_tecnica="Corrientes esmeralda muy fuertes. Combinar spinning con señuelos pesados de colores vivos y moscas con sink tip rápido."),

            # 15. Región de Aysén
            PuntoPescaChile(nombre="Río Baker (Confluencia Nef)", region="Región de Aysén", latitud=-47.020, longitud=-72.820, tipo_agua="Dulce",
                            especies_objetivo="Trucha Arcoíris trofeo, Trucha Fario", recomendacion_tecnica="Río extremadamente caudaloso de color turquesa. Imprescindible pescar remansos (eddies) con ninfas grandes o rapalas pesadas."),
            PuntoPescaChile(nombre="Lago General Carrera (Puerto Tranquilo)", region="Región de Aysén", latitud=-46.620, longitud=-72.670, tipo_agua="Dulce",
                            especies_objetivo="Trucha Fario gigante, Trucha Arcoíris", recomendacion_tecnica="Fuertes vientos patagónicos. Lanzar chispas compactas o hacer trolling costero con señuelos articulados imitando puyes."),

            # 16. Región de Magallanes
            PuntoPescaChile(nombre="Río Serrano (PN Torres del Paine)", region="Región de Magallanes", latitud=-51.240, longitud=-73.050, tipo_agua="Dulce",
                            especies_objetivo="Salmón Chinook, Trucha Fario", recomendacion_tecnica="Escenario mundial de pesca. Tramos de corriente lenta y pozones profundos. Uso de cucharillas gigantes y moscas Intruder."),
            PuntoPescaChile(nombre="Río Grande / Río Penitente (Tierra del Fuego)", region="Región de Magallanes", latitud=-53.600, longitud=-69.200, tipo_agua="Dulce",
                            especies_objetivo="Trucha Fario Sea Run (Anádroma)", recomendacion_tecnica="Truchas migratorias gigantes. Vientos extremos: usar cañas de dos manos (Spey) y streamers con patas de goma en curvas socavadas.")
        ]
        db.bulk_save_objects(puntos_base)
        db.commit()
        print(
            f"Agent Log: {len(puntos_base)} puntos de pesca cargados exitosamente.")
    db.close()


def serializar_especiales(obj):
    nombre_clase = type(obj).__name__

    if nombre_clase == 'Decimal':
        return float(obj)
    if nombre_clase in ['datetime', 'date', 'time', 'Timestamp']:
        return obj.isoformat()

    # Si llega algún otro dato raro de PostgreSQL (como un UUID o JSONB),
    # lo convertimos a texto plano en lugar de lanzar un error.
    return str(obj)


def ejecutar_sql_sandbox(query: str):
    print(f"🔍 Ejecutando SQL generado por IA: {query}")
    try:
        # 1. Conexión usando EXCLUSIVAMENTE el usuario restringido
        conn = psycopg2.connect(os.getenv("SUPABASE_IA_DB_URL"))
        cursor = conn.cursor()

        cursor.execute(query)

        # 2. Protección de tokens: extraemos máximo 50 filas
        resultados = cursor.fetchmany(50)

        # 3. Extraemos los nombres de las columnas para que el JSON tenga sentido
        nombres_columnas = [desc[0] for desc in cursor.description]
        datos = [dict(zip(nombres_columnas, fila)) for fila in resultados]

        cursor.close()
        conn.close()

        # Si la consulta fue exitosa pero no hay datos
        if not datos:
            return "No se encontraron registros para esta consulta."

        # Usamos el serializador personalizado para procesar Decimals y Datetimes de forma segura
        return json.dumps(datos, default=serializar_especiales)

    except Exception as e:
        return "Error consultando las estadísticas de la base de datos."


inicializar_conocimiento_pesca()
inicializar_puntos_pesca()

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
    },
    {
        "type": "function",
        "function": {
            "name": "consultar_puntos_pesca",
            "description": "EJECUTA ESTA HERRAMIENTA SOLO si el usuario pregunta por lugares para pescar, dónde ir, picadas destacadas o pide recomendaciones para pescar en cierto lugar de Chile.",
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

            print(
                f"DEBUG VISION - ID: {clase_id}, Nombre: {nombre_clase}, Confianza: {confianza}")

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
        "- 'abrir_seccion_nudos': Para enseñar a atar nudos.\n"
        "- 'consultar_puntos_pesca': Para recomendar lugares de pesca o cuando el usuario pida recomendaciones para pescar en un lugar específico.\n"
        "Si la consulta es de otro tipo responde con tus conocimientos."
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
                elif func_name == "consultar_puntos_pesca":
                    resultado = consultar_puntos_pesca_db()
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
                        "abrir_seccion_nudos", "sección de nudos")

                elif "consultar_regulaciones_especie" in raw_content:
                    datos = consultar_regulaciones_db()
                    raw_content = re.sub(
                        r'<tool_call>.*?</tool_call>', f'Consultando la base de datos...\n\n{datos}', raw_content, flags=re.DOTALL)
                    raw_content = raw_content.replace(
                        "consultar_regulaciones_especie", "regulaciones oficiales")

                elif "consultar_bitacora" in raw_content:
                    datos = consultar_bitacora_db()
                    raw_content = re.sub(
                        r'<tool_call>.*?</tool_call>', f'Revisando tu historial...\n\n{datos}', raw_content, flags=re.DOTALL)
                    raw_content = raw_content.replace(
                        "consultar_bitacora", "bitácora")

                elif "consultar_puntos_pesca" in raw_content or "puntos_pesca" in raw_content:
                    datos = consultar_puntos_pesca_db()
                    raw_content = re.sub(
                        r'<tool_call>.*?</tool_call>', f'Consultando el atlas GIS offline...\n\n{datos}', raw_content, flags=re.DOTALL)
                    raw_content = raw_content.replace(
                        "consultar_puntos_pesca", "puntos de pesca destacados")

        clean_content = re.sub(r'<think>.*?</think>\n*',
                               '', raw_content, flags=re.DOTALL).strip()

        return {
            "role": "assistant",
            "content": clean_content,
            "action": action_flag
        }

    except Exception as e:
        return {"role": "assistant", "content": f"Error conectando al LLM: {str(e)}"}

herramientas_orquestador = [
    {
        "type": "function",
        "function": {
            "name": "enrutar_a_agente_clima",
            "description": "Invocar UNICAMENTE cuando el usuario pregunte por el clima, mareas, viento, fases lunares o condiciones meteorológicas.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ubicacion": {
                        "type": "string",
                        "description": "Lugar de Chile por el que pregunta el usuario. Ej: 'Curanipe', 'Río Toltén'."
                    },
                    "solicitud_especifica": {
                        "type": "string",
                        "description": "La duda exacta sobre el clima."
                    }
                },
                "required": ["ubicacion", "solicitud_especifica"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "enrutar_a_agente_analista",
            "description": "Invocar UNICAMENTE cuando el usuario pregunte por historiales de captura, estadísticas en la nube, leaderboards o tamaños récord.",
            "parameters": {
                "type": "object",
                "properties": {
                    "detalle_consulta": {
                        "type": "string",
                        "description": "El objetivo de la búsqueda. Ej: 'Récord de Corvina'."
                    }
                },
                "required": ["detalle_consulta"]
            }
        }
    }
]


@app.post("/chat-cloud")
async def chat_cloud_endpoint(request: ChatRequest):
    if not orquestador_client:
        return {"role": "assistant", "content": "Error: El Orquestador no está configurado."}

    # Inyectar el System Prompt del Orquestador
    mensajes_cloud = [
        {
            "role": "system",
            "content": "Eres el Orquestador Central de una app de pesca en Chile. Responde de forma directa, técnica y usa el sistema métrico. REGLAS DE ENRUTAMIENTO ESTRICTAS: 1. Si preguntan del clima, mareas o viento, DEBES usar la herramienta 'enrutar_a_agente_clima'. 2. Si preguntan por estadísticas, leaderboards o su historial en la nube, DEBES usar la herramienta 'enrutar_a_agente_analista'. Para dudas generales de pesca (nudos, equipos, técnicas), responde tú mismo."
        }
    ]

    # Construir el historial para dar contexto al modelo
    for msg in request.messages:
        if msg.role == "assistant" and ("¡Hola!" in msg.content or "📸" in msg.content):
            continue
        mensajes_cloud.append({"role": msg.role, "content": msg.content})

    # Inyección de contexto visual (Señuelo actual detectado por YOLO)
    if request.senuelo_actual not in ["Ninguno", "desconocido", "Analizando..."]:
        mensajes_cloud[-1]["content"] += f"\n\n[INFO DEL SISTEMA: El usuario tiene un señuelo '{request.senuelo_actual}' en la cámara.]"

    try:
        response = orquestador_client.chat.completions.create(
            model=deployment_name,
            messages=mensajes_cloud,
            tools=herramientas_orquestador,
            tool_choice="auto",
            temperature=0.3
        )

        mensaje_respuesta = response.choices[0].message

        # INTERCEPTOR DE HERRAMIENTAS (Function Calling)
        if mensaje_respuesta.tool_calls:
            tool_call = mensaje_respuesta.tool_calls[0]
            nombre_funcion = tool_call.function.name
            import json
            argumentos = json.loads(tool_call.function.arguments)

            if nombre_funcion == "enrutar_a_agente_clima":
                print(f"✅ ORQUESTADOR DELEGA -> AGENTE CLIMA: {argumentos}")
                return {
                    "role": "assistant",
                    "content": f"*(Conectando con el satélite meteorológico para revisar {argumentos.get('ubicacion', 'la zona')}...)*",
                    "action": "route_clima",
                    "parametros_agente": argumentos
                }

            elif nombre_funcion == "enrutar_a_agente_analista":
                consulta_usuario = argumentos.get('detalle_consulta', '')
                print(
                    f"✅ ORQUESTADOR DELEGA -> AGENTE ANALISTA: {consulta_usuario}")

                try:
                    # PASO 1: El Agente Analista (Text-to-SQL) genera la consulta
                    res_analista = agentes_client.responses.create(
                        input=[{"role": "user", "content": consulta_usuario}],
                        extra_body={
                            "agent_reference": {
                                # Asegúrate de que este sea el nombre de tu agente en Foundry
                                "name": "agente-analista",
                                "version": "6",
                                "type": "agent_reference"
                            }
                        }
                    )

                    sql_crudo = res_analista.output_text.strip()

                    # Limpiamos el texto por si el modelo agregó formato markdown (```sql ... ```)
                    sql_crudo = sql_crudo.replace(
                        "```sql", "").replace("```", "").strip()

                    # PASO 2: Ejecutamos el SQL en la base de datos blindada
                    resultado_json = ejecutar_sql_sandbox(sql_crudo)

                    # PASO 3: El Orquestador traduce el JSON frío a un comentario experto
                    traduccion = orquestador_client.chat.completions.create(
                        model=deployment_name,
                        messages=[
                            {"role": "system", "content": "Eres un asistente de pesca. Recibes una pregunta y un resultado JSON crudo de la base de datos. Traduce los datos a una respuesta natural, directa y amigable. Si hay un error, dile al usuario que la bitácora no tiene esos datos."},
                            {"role": "user", "content": f"Pregunta: {consulta_usuario}\nDatos DB: {resultado_json}"}
                        ],
                        temperature=0.3
                    )

                    respuesta_final = traduccion.choices[0].message.content

                    return {
                        "role": "assistant",
                        "content": respuesta_final,
                        "action": None
                    }

                except Exception as e:
                    print(f"❌ Error en el flujo del Analista:")
                    return {
                        "role": "assistant",
                        "content": "Estuve revisando la bitácora, pero tuve un problema de conexión con los registros. Intentémoslo de nuevo más tarde."
                    }

        # RESPUESTA DIRECTA (Si es una pregunta general)
        return {
            "role": "assistant",
            "content": mensaje_respuesta.content,
            "action": None
        }

    except Exception as e:
        print(f"Error en el orquestador:")
        return {"role": "assistant", "content": f"Error conectando a Azure"}


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


@app.get("/puntos-pesca")
async def obtener_puntos_pesca(db: Session = Depends(get_db)):
    """
    Devuelve la lista completa de puntos estratégicos GIS.
    El frontend web puede solicitar esta ruta al cargar la vista del mapa
    para colocar todos los pines y filtrar de forma local mediante React.
    """
    return db.query(PuntoPescaChile).all()


@app.post("/agente-clima")
async def invocar_agente_clima(req: ParametrosClima):
    if not agentes_client:
        raise HTTPException(
            status_code=500,
            detail="Falta configuración del Agente o credenciales de Entra ID."
        )

    print(
        f"🌦️ COMIENZA EVALUACIÓN AGENTIC: Despertando Agente Clima para {req.ubicacion}")

    try:
        # Llamada directa y sin estado a tu agente hospedado
        response = agentes_client.responses.create(
            input=[
                {"role": "user", "content": f"Revisa el clima para: {req.ubicacion}. La duda puntual es: {req.solicitud_especifica}"}
            ],
            extra_body={
                "agent_reference": {
                    "name": "agente-clima",
                    "version": "2",
                    "type": "agent_reference"
                }
            }
        )

        # La respuesta ya viene empaquetada lista para usar
        respuesta_final = response.output_text

        print(f"✅ RESPUESTA AGENTE CLIMA GENERADA EXITOSAMENTE")
        return {"respuesta": respuesta_final}

    except Exception as e:
        print(f"❌ Error crítico en Responses API: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error de comunicación con Azure Responses Service."
        )
