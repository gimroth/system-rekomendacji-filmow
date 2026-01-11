from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

from app.database import Base, engine
# WAŻNE: Importujemy modele (user i comment), aby SQLAlchemy wiedziało, że ma utworzyć dla nich tabele
from app.models import user, comment 
from app.routers import auth, admin, preferences, movies, comments
from app.dependencies import get_current_admin
from app.models.user import User # Import potrzebny do mockowania admina poniżej

# Ta linia tworzy tabele w bazie danych (jeśli jeszcze nie istnieją)
Base.metadata.create_all(bind=engine)

templates = Jinja2Templates(directory="templates")
app = FastAPI(title="System Rekomendacji Filmów")

# Konfiguracja CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pliki statyczne (CSS, JS, obrazy)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Rejestracja Routerów (endpointów API)
app.include_router(auth.router)
app.include_router(admin.router)
app.include_router(preferences.router)
app.include_router(movies.router)
app.include_router(comments.router) # Router komentarzy

# --- MOCK ADMINA (Do celów developerskich) ---
def mock_get_current_admin():
    # Udajemy, że zawsze jest zalogowany admin
    # UWAGA: Usuń to lub zakomentuj, gdy będziesz testować prawdziwe logowanie admina!
    return User(id=1, username="DevAdmin", email="admin@dev.com", role="admin")

app.dependency_overrides[get_current_admin] = mock_get_current_admin
# ---------------------------------------------

# --- ENDPOINTY HTML (Frontend rendering) ---

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    # Przekierowanie na stronę główną lub index
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    # Zakładam, że masz register.html, jeśli nie - użyj index.html
    return templates.TemplateResponse("register.html", {"request": request})

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