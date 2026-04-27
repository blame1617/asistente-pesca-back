from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

# La URL de conexión para SQLite
SQLALCHEMY_DATABASE_URL = "sqlite:///./pesca.db"

# El motor (engine) es el que se encarga de hablar con el archivo .db
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# La sesión es la que usaremos para hacer las consultas
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Clase base de la que heredarán nuestros modelos
Base = declarative_base()

# Definición de la Tabla de Capturas


class Captura(Base):
    __tablename__ = "capturas"

    id = Column(Integer, primary_key=True, index=True)
    especie = Column(String)
    medida_cm = Column(Float)
    senuelo = Column(String)
    fecha = Column(DateTime, default=datetime.datetime.utcnow)
    ruta_imagen = Column(String)

# --- NUEVA TABLA DE CONOCIMIENTO TÉCNICO (REGULACIONES) ---


class EspecieChile(Base):
    __tablename__ = "especies_chile"

    id = Column(Integer, primary_key=True, index=True)
    nombre = Column(String, unique=True, index=True)
    zona = Column(String)
    tipo_agua = Column(String)
    senuelos = Column(String)
    regulacion = Column(String)


# Esta línea crea el archivo pesca.db y la tabla si no existen
Base.metadata.create_all(bind=engine)
