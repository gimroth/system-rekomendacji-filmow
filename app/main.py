from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from app.database import Base, engine
from app.routers import auth, admin, preferences
from app.models.user import User
from app.dependencies import get_current_admin

Base.metadata.create_all(bind=engine)

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

def mock_get_current_admin():
    # Udajemy, że zawsze jest zalogowany admin
    return User(id=1, username="DevAdmin", email="admin@dev.com", role="admin")

app.dependency_overrides[get_current_admin] = mock_get_current_admin

@app.get("/login", response_class=HTMLResponse)
async def login_page():
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse("<h1>Brak pliku index.html</h1>", status_code=404)

@app.get("/admin-panel", response_class=HTMLResponse)
async def admin_page():
    try:
        with open("static/admin.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse("<h1>Brak pliku static/admin.html</h1>", status_code=404)

@app.get("/onboarding", response_class=HTMLResponse)
async def preferences_page():
    try:
        with open("static/preferences.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Błąd: Plik preferences.html nie znaleziony!</h1>", status_code=404)

@app.get("/")
def root():
    return {"message": "API działa! Przejdź do /admin-panel"}