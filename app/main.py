from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

# Import bazy danych
from app.database import Base, engine

# Import modeli (żeby SQLAlchemy wiedziało co stworzyć w bazie)
# Tutaj łączymy Twoje modele i ewentualne modele dziewczyn
from app.models import user, comment, preference, movie

# Import Twoich routerów (LOGIKA)
from app.routers import auth, admin, preferences, movies, comments

# Tworzenie tabel w bazie
Base.metadata.create_all(bind=engine)

app = FastAPI(title="System Rekomendacji Filmów")

# Konfiguracja CORS (bezpieczeństwo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Obsługa plików statycznych (CSS, obrazki)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Konfiguracja szablonów HTML
templates = Jinja2Templates(directory="templates")

# --- PODPINANIE ROUTERÓW (API) ---
# To sprawia, że Twoje logowanie i komentarze działają
app.include_router(auth.router)
app.include_router(admin.router)
app.include_router(preferences.router)
app.include_router(movies.router)
app.include_router(comments.router)


# --- ENDPOINTY HTML (WIDOKI) ---
# To są strony, które dodały koleżanki

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    # Strona startowa
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