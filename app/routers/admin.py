from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, or_
from typing import List, Optional
import requests
import os
from app.database import get_db
from app.dependencies import get_current_admin
from app.models.user import User
from app.models.movie import Movie
from app.models.rating import Rating
from app.schemas.admin_schemas import (
    MovieAdminResponse,
    TMDBSearchResult,
    MovieStats,
    PaginatedUsersResponse
)

try:
    from app.config import settings

    TMDB_API_KEY = settings.TMDB_API_KEY
except ImportError:
    TMDB_API_KEY = os.getenv("TMDB_API_KEY")

router = APIRouter(prefix="/admin", tags=["Admin Dashboard"])

@router.get("/users", response_model=PaginatedUsersResponse)
def get_all_users(
        skip: int = 0,
        limit: int = 20,
        search: Optional[str] = None,
        db: Session = Depends(get_db),
        admin: User = Depends(get_current_admin)
):
    query = db.query(User)

    if search:
        search_filter = f"%{search}%"
        query = query.filter(
            or_(
                User.username.ilike(search_filter),
                User.email.ilike(search_filter)
            )
        )

    total_users = query.count()
    users = query.offset(skip).limit(limit).all()

    return {"total": total_users, "users": users}

@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(user_id: int, db: Session = Depends(get_db), admin: User = Depends(get_current_admin)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Użytkownik nie znaleziony")

    if user.role == 'admin':
        raise HTTPException(status_code=400, detail="Nie można usunąć administratora.")

    db.delete(user)
    db.commit()
    return

@router.get("/movies/search", response_model=List[TMDBSearchResult])
def search_movies_in_tmdb(query: str, admin: User = Depends(get_current_admin)):
    """Wyszukuje filmy w TMDB (Język: ANGIELSKI)."""
    if not query:
        return []

    if not TMDB_API_KEY:
        raise HTTPException(status_code=500, detail="Brak klucza API TMDB w konfiguracji.")

    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={query}&language=en-US"

    try:
        response = requests.get(url, timeout=10)  # Timeout zapobiega zawieszeniu
    except requests.RequestException:
        raise HTTPException(status_code=502, detail="Błąd połączenia z TMDB")

    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Błąd TMDB: {response.status_code}")

    results = response.json().get("results", [])
    parsed_results = []

    for item in results:
        release_date = item.get("release_date", "")
        year = release_date[:4] if release_date else None

        parsed_results.append(TMDBSearchResult(
            tmdb_id=item["id"],
            title=item["title"],
            year=year,
            poster_url=f"https://image.tmdb.org/t/p/w200{item.get('poster_path')}" if item.get("poster_path") else None,
            overview=item.get("overview")
        ))

    return parsed_results

@router.post("/movies/tmdb/{tmdb_id}", response_model=MovieAdminResponse)
def add_movie_by_tmdb_id(tmdb_id: int, db: Session = Depends(get_db), admin: User = Depends(get_current_admin)):
    """Dodaje film do bazy lokalnej na podstawie ID z TMDB (Język: ANGIELSKI)."""

    if db.query(Movie).filter(Movie.tmdb_id == tmdb_id).first():
        raise HTTPException(status_code=400, detail="Ten film już istnieje w bazie!")

    if not TMDB_API_KEY:
        raise HTTPException(status_code=500, detail="Brak klucza API TMDB.")

    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}&language=en-US"

    try:
        response = requests.get(url, timeout=10)
    except requests.RequestException:
        raise HTTPException(status_code=502, detail="Błąd połączenia z TMDB")

    if response.status_code != 200:
        raise HTTPException(status_code=404, detail="Nie znaleziono filmu w TMDB")

    data = response.json()

    release_date = data.get("release_date", "")
    year_val = None
    if release_date and len(release_date) >= 4:
        try:
            year_val = int(release_date[:4])
        except ValueError:
            year_val = None

    new_movie = Movie(
        title=data.get("title"),
        year=year_val,
        tmdb_id=tmdb_id,
        description=data.get("overview"),
        poster_url=f"https://image.tmdb.org/t/p/w500{data.get('poster_path')}" if data.get("poster_path") else None
    )

    try:
        db.add(new_movie)
        db.commit()
        db.refresh(new_movie)
    except Exception as e:
        db.rollback()
        # Wypisz błąd w konsoli serwera, żebyś widziała co się stało
        print(f"BŁĄD BAZY DANYCH: {e}")
        raise HTTPException(status_code=500, detail="Błąd zapisu do bazy danych. Sprawdź logi serwera.")

    return new_movie


@router.get("/movies/local", response_model=List[MovieStats])
def get_local_movies(
        skip: int = 0,
        limit: int = 50,
        search: Optional[str] = None,
        db: Session = Depends(get_db),
        admin: User = Depends(get_current_admin)
):
    query = db.query(
        Movie.id,
        Movie.title,
        Movie.year,
        Movie.tmdb_id,
        func.count(Rating.id).label("ratings_count"),
        func.coalesce(func.avg(Rating.rating), 0.0).label("average_rating")
    ).outerjoin(Rating, Movie.id == Rating.movie_id)

    if search:
        query = query.filter(Movie.title.ilike(f"%{search}%"))

    query = query.group_by(Movie.id).order_by(desc("ratings_count"), desc(Movie.id))
    results = query.offset(skip).limit(limit).all()

    return [
        MovieStats(
            id=row.id,
            title=row.title,
            year=row.year,
            tmdb_id=row.tmdb_id,
            ratings_count=row.ratings_count,
            average_rating=round(float(row.average_rating), 2)
        )
        for row in results
    ]

@router.delete("/movies/{movie_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_movie(movie_id: int, db: Session = Depends(get_db), admin: User = Depends(get_current_admin)):
    movie = db.query(Movie).filter(Movie.id == movie_id).first()
    if not movie:
        raise HTTPException(status_code=404, detail="Film nie znaleziony")
    db.delete(movie)
    db.commit()
    return