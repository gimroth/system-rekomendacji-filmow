from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager

# --- IMPORTY BAZY DANYCH ---
from app.database import Base, engine
# Importujemy modele, aby SQLAlchemy utworzyło tabele
from app.models import user, comment, movie, rating

# --- IMPORT MODELU ML ---
from app.ml.loader import load_model

# --- IMPORTY ROUTERÓW ---

from app.routers import auth, admin, preferences, movies, comments, recommendations, ratings, homepage
from app.routers import users
# Tworzenie tabel w bazie (jeśli nie istnieją)
Base.metadata.create_all(bind=engine)

# --- CYKL ŻYCIA APLIKACJI (LIFESPAN) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Kod uruchamiany przy starcie serwera
    load_model()
    yield
    # Kod uruchamiany przy zamknięciu (opcjonalnie)

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

# Konfiguracja szablonów
templates = Jinja2Templates(directory="templates")

# --- PODPINANIE ROUTERÓW API ---
app.include_router(auth.router)
app.include_router(admin.router)
app.include_router(preferences.router)
app.include_router(movies.router)
app.include_router(comments.router)
app.include_router(recommendations.router)
app.include_router(ratings.router)
app.include_router(homepage.router)
app.include_router(users.router) # <--- DODANO ROUTER UŻYTKOWNIKÓW

# --- WIDOKI HTML ---
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/home", response_class=HTMLResponse)
async def home_page(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/rankings", response_class=HTMLResponse)
async def rankings_page(request: Request):
    return templates.TemplateResponse("rankings.html", {"request": request})

@app.get("/movie", response_class=HTMLResponse)
async def movie_detail_page(request: Request):
    return templates.TemplateResponse("movie.html", {"request": request})

@app.get("/admin-panel", response_class=HTMLResponse)
async def admin_page(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})

@app.get("/onboarding", response_class=HTMLResponse)
async def preferences_page(request: Request):
    return templates.TemplateResponse("preferences.html", {"request": request})

@app.get("/my-recommendations", response_class=HTMLResponse)
async def my_recommendations_page(request: Request):
    return templates.TemplateResponse("recommendations.html", {"request": request})

# DODANO: Endpoint dla profilu
@app.get("/profile", response_class=HTMLResponse)
async def profile_page(request: Request):
    return templates.TemplateResponse("profile.html", {"request": request})