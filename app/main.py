from fastapi import FastAPI
from app.database import Base, engine
from app.routers import auth
from app.routers import admin  # Dodano import admin routera
from fastapi.staticfiles import StaticFiles
from starlette.responses import HTMLResponse
from app.routers import preferences

# Tworzy tabele
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Movie Recommendation API") # Dodano title

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/login", response_class=HTMLResponse)
async def login_page():
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Błąd: Plik index.html nie znaleziony!</h1>", status_code=404)

@app.get("/onboarding", response_class=HTMLResponse)
async def preferences_page():
    try:
        with open("static/preferences.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Błąd: Plik preferences.html nie znaleziony!</h1>", status_code=404)

app.include_router(auth.router)
app.include_router(admin.router) # Dodano admin router
app.include_router(preferences.router)

@app.get("/")
def root():
    return {"message": "API działa!"}