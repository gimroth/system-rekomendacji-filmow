import os
from pathlib import Path
from typing import Optional, List
from jose import jwt, JWTError
from passlib.context import CryptContext
from pydantic import BaseModel

from fastapi import APIRouter, Request, Depends, HTTPException, status
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import desc, func

from app.database import get_db
from app.models import Movie, Rating, User, Genre

# Konfiguracja hashowania
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = "bardzo_tajny_klucz"
ALGORITHM = "HS256"

router = APIRouter()

# --- KONFIGURACJA SZABLONÓW ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
templates_dir = BASE_DIR / "templates"
if not templates_dir.exists():
    templates_dir = BASE_DIR / "app" / "templates"

templates = Jinja2Templates(directory=str(templates_dir))


# --- POMOCNIK: WYCIĄGANIE USERA Z CIASTECZKA ---
def get_user_from_cookie(request: Request, db: Session) -> Optional[User]:
    token = request.cookies.get("access_token")
    if not token:
        return None
    try:
        if token.startswith("Bearer "):
            token = token.split(" ")[1]
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
        return db.query(User).filter(User.username == username).first()
    except JWTError:
        return None


# ==========================================
# 1. WIDOKI HTML
# ==========================================

@router.get("/")
async def home(request: Request, db: Session = Depends(get_db)):
    user = get_user_from_cookie(request, db)
    return templates.TemplateResponse("index.html", {"request": request, "user": user})

@router.get("/login")
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@router.get("/register")
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@router.get("/rankings")
async def rankings_page(request: Request, db: Session = Depends(get_db)):
    user = get_user_from_cookie(request, db)
    return templates.TemplateResponse("rankings.html", {"request": request, "user": user})

@router.get("/profile")
async def profile_page(request: Request, db: Session = Depends(get_db)):
    user = get_user_from_cookie(request, db)
    if not user:
        return templates.TemplateResponse("login.html", {"request": request})
    return templates.TemplateResponse("profile.html", {"request": request, "user": user})

@router.get("/logout")
async def logout(request: Request):
    response = templates.TemplateResponse("login.html", {"request": request})
    response.delete_cookie("access_token")
    return response

@router.get("/movie/{movie_id}")
async def movie_details_page(request: Request, movie_id: int, db: Session = Depends(get_db)):
    user = get_user_from_cookie(request, db)
    return templates.TemplateResponse("movie_details.html", {
        "request": request, 
        "movie_id": movie_id,
        "user": user
    })


# ==========================================
# 2. API DLA STRONY GŁÓWNEJ
# ==========================================

@router.get("/homepage/latest")
def get_latest_movies(limit: int = 20, db: Session = Depends(get_db)):
    movies = db.query(Movie).order_by(Movie.year.desc()).limit(limit).all()
    return [{
        "id": m.id, "title": m.title, "year": m.year, "poster_url": m.poster_url,
        "genres": [{"name": g.name} for g in m.genres] if m.genres else []
    } for m in movies]

@router.get("/homepage/top-by-aspect")
def get_top_by_aspect(aspect: str, limit: int = 20, db: Session = Depends(get_db)):
    query = db.query(Movie, func.coalesce(func.avg(Rating.rating), 0.0).label("avg_rating"))\
             .outerjoin(Rating, Movie.id == Rating.movie_id).group_by(Movie.id)\
             .having(func.count(Rating.id) > 0).order_by(desc("avg_rating")).limit(limit)
    results = query.all()
    return [{
        "id": row[0].id, "title": row[0].title, "poster_url": row[0].poster_url,
        "rating": row[1], "year": row[0].year
    } for row in results]


# ==========================================
# 3. OBSŁUGA PROFILU (DOSTOSOWANA DO MODELU)
# ==========================================

class ProfileUpdate(BaseModel):
    user_id: int
    new_username: str
    new_email: str

class PasswordUpdate(BaseModel):
    user_id: int
    old_password: str
    new_password: str

@router.get("/api/user-ratings/{user_id}")
def get_user_ratings_history(user_id: int, db: Session = Depends(get_db)):
    ratings = db.query(Rating).filter(Rating.user_id == user_id).order_by(Rating.timestamp.desc()).all()
    results = []
    for r in ratings:
        movie = db.query(Movie).filter(Movie.id == r.movie_id).first()
        if movie:
            results.append({
                "movie_title": movie.title,
                "rating": r.rating,
                "timestamp": r.timestamp.strftime("%Y-%m-%d") if r.timestamp else "",
                "movie_id": movie.id,
                "poster_url": movie.poster_url
            })
    return results

@router.put("/api/profile/update")
def update_profile_data(data: ProfileUpdate, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == data.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Użytkownik nie istnieje")
    
    existing = db.query(User).filter(
        ((User.username == data.new_username) | (User.email == data.new_email)) & (User.id != data.user_id)
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Nazwa lub email zajęte")

    user.username = data.new_username
    user.email = data.new_email
    db.commit()
    return {"message": "OK"}

@router.put("/api/profile/password")
def change_user_password(data: PasswordUpdate, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == data.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Użytkownik nie istnieje")

    # POPRAWKA: Używamy password_hash zamiast hashed_password
    if not pwd_context.verify(data.old_password, user.password_hash):
        raise HTTPException(status_code=400, detail="Stare hasło nieprawidłowe")

    user.password_hash = pwd_context.hash(data.new_password) # <-- TUTAJ TEŻ ZMIANA
    db.commit()
    return {"message": "OK"}