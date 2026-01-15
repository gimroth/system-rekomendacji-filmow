from fastapi import FastAPI, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

# --- IMPORTY ZALEŻNOŚCI ---
from app.dependencies import get_current_admin

# --- IMPORTY MODELI (Baza Danych) ---
from app.database import Base, engine
# Dodajemy 'rating' do importów, aby baza utworzyła tabelę 'ratings'
from app.models import user, comment, movie, rating

# --- IMPORTY ROUTERÓW ---
# Dodajemy 'ratings' do listy routerów
from app.routers import auth, admin, preferences, movies, comments, recommendations, ratings

# Tworzenie tabel w bazie (jeśli nie istnieją)
Base.metadata.create_all(bind=engine)

app = FastAPI(title="System Rekomendacji Filmów")

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
app.include_router(ratings.router)  # <--- NOWOŚĆ: Router ocen

# --- ENDPOINTY HTML (WIDOKI) ---

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