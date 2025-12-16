from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from app.database import Base, engine
from app.routers import auth, admin, preferences, movies
from app.models.user import User
from app.dependencies import get_current_admin
from fastapi import Request
from fastapi.templating import Jinja2Templates

Base.metadata.create_all(bind=engine)

templates = Jinja2Templates(directory="templates")
app = FastAPI(title="System Rekomendacji Filmów")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(auth.router)
app.include_router(admin.router)
app.include_router(preferences.router)
app.include_router(movies.router)

def mock_get_current_admin():
    # Udajemy, że zawsze jest zalogowany admin
    return User(id=1, username="DevAdmin", email="admin@dev.com", role="admin")

app.dependency_overrides[get_current_admin] = mock_get_current_admin

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
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
    # Admin ma swój navbar wewnątrz pliku, więc może zostać jak jest,
    # albo też możesz go przenieść do templates.
    return templates.TemplateResponse("admin.html", {"request": request})

@app.get("/onboarding", response_class=HTMLResponse)
async def preferences_page(request: Request):
    return templates.TemplateResponse("preferences.html", {"request": request})

@app.get("/")
def root():
    return {"message": "API działa!"}