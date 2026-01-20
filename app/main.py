from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# --- IMPORTY BAZY DANYCH ---
from app.database import Base, engine
from app.models import user, comment, movie, rating

# --- IMPORT MODELU ML ---
from app.ml.anfis_model import load_anfis_model

# --- IMPORTY ROUTERÓW ---
from app.routers import auth, admin, preferences, movies, comments, recommendations, ratings
from app.routers import homepage

# Tworzenie tabel w bazie
Base.metadata.create_all(bind=engine)

# --- CYKL ŻYCIA APLIKACJI ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_anfis_model() 
    yield

# --- INICJALIZACJA APLIKACJI ---
app = FastAPI(
    title="System Rekomendacji Filmów",
    lifespan=lifespan
)

# Konfiguracja CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Obsługa plików statycznych
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- PODPINANIE ROUTERÓW API ---

# 1. AUTH - KLUCZOWA POPRAWKA: prefix="/auth"
# To sprawia, że adres logowania to: http://127.0.0.1:8000/auth/token
app.include_router(auth.router, prefix="/auth", tags=["auth"])

# 2. Reszta routerów API
app.include_router(admin.router, prefix="/admin", tags=["admin"])
app.include_router(preferences.router, prefix="/preferences", tags=["preferences"])
app.include_router(movies.router, prefix="/movies", tags=["movies"])
app.include_router(comments.router, prefix="/comments", tags=["comments"])
app.include_router(recommendations.router, prefix="/recommendations", tags=["recommendations"])
app.include_router(ratings.router, prefix="/ratings", tags=["ratings"])

# 3. HOMEPAGE - Obsługa HTML (Login, Register, Profil, Home)
# Ten router musi być podpięty, bo w nim jest logika profilu użytkownika
app.include_router(homepage.router)

