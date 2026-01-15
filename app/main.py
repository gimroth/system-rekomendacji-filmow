from fastapi import FastAPI, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

# --- IMPORTY ZALEŻNOŚCI ---
# Od dziewczyn:
from app.dependencies import get_current_admin

# Baza danych i modele (Twoje - KLUCZOWE):
from app.database import Base, engine
from app.models import user, comment, movie

# --- IMPORTY ROUTERÓW ---
# Łączymy routery Twoje (auth, admin, movies, comments) i dziewczyn (preferences, recommendations)
from app.routers import auth, admin, preferences, movies, comments, recommendations

# Tworzenie tabel w bazie (musi być po importach modeli)
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
app.include_router(comments.router)       # Twój router
app.include_router(recommendations.router) # Router od dziewczyn

# --- ENDPOINTY HTML (WIDOKI) ---

# Strona główna - zostawiamy Twoją wersję HTML (lepsza dla użytkownika niż JSON)
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

# --- NOWE WIDOKI OD DZIEWCZYN ---
@app.get("/onboarding", response_class=HTMLResponse)
async def preferences_page(request: Request):
    return templates.TemplateResponse("preferences.html", {"request": request})

@app.get("/my-recommendations", response_class=HTMLResponse)
async def my_recommendations_page(request: Request):
    return templates.TemplateResponse("recommendations.html", {"request": request})