from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, case
from typing import List, Optional
import requests
import os

from app.database import get_db
from app.models.movie import Movie
from app.models.rating import Rating
from app.models.genre import Genre, MovieGenre

# Pobranie klucza API
try:
    from app.config import settings

    TMDB_API_KEY = settings.TMDB_API_KEY
except ImportError:
    TMDB_API_KEY = os.getenv("TMDB_API_KEY")

router = APIRouter(prefix="/movies", tags=["Movies (Public)"])


# 1. Główna lista filmów (Z "Miękkim" rankingiem)
@router.get("/", response_model=dict)
def get_movies(
        page: int = 1,
        limit: int = 12,
        search: Optional[str] = None,
        genre: Optional[str] = None,
        min_count: int = 0,  # <--- To będzie nasz próg priorytetu (np. 500)
        db: Session = Depends(get_db)
):
    skip = (page - 1) * limit

    # Budujemy zapytanie
    query = db.query(
        Movie.id,
        Movie.title,
        Movie.year,
        Movie.poster_url,
        Movie.tmdb_id,
        func.count(Rating.id).label("ratings_count"),
        func.coalesce(func.avg(Rating.rating), 0.0).label("average_rating")
    ).outerjoin(Rating, Movie.id == Rating.movie_id)

    # Filtracja (Search / Gatunek)
    if search:
        query = query.filter(Movie.title.ilike(f"%{search}%"))

    if genre and genre != "Wszystkie":
        query = query.join(Movie.movie_genres).join(MovieGenre.genre).filter(Genre.name == genre)

    query = query.group_by(Movie.id)

    # --- LOGIKA SORTOWANIA PRIORYTETOWEGO ---
    # Jeśli min_count > 0, tworzymy "flagę":
    # 1 = Film ma wystarczająco dużo głosów (Priorytet)
    # 0 = Film ma za mało głosów (Reszta)

    if min_count > 0:
        priority_sort = case(
            (func.count(Rating.id) >= min_count, 1),  # Warunek spełniony -> daj 1
            else_=0  # Warunek niespełniony -> daj 0
        )
        # Sortujemy:
        # 1. Najpierw wg priorytetu (grupa > 500 głosów będzie wyżej)
        # 2. Potem wg oceny (wewnątrz każdej grupy)
        # 3. Potem wg liczby głosów (żeby w grupie "słabszej" te popularniejsze były wyżej)
        query = query.order_by(desc(priority_sort), desc("average_rating"), desc("ratings_count"))
    else:
        # Standardowe sortowanie jeśli nie podano progu
        query = query.order_by(desc("average_rating"), desc("ratings_count"))

    # Pobieramy wyniki
    total = query.count()
    movies = query.offset(skip).limit(limit).all()

    # Formatowanie wyniku
    results = []
    for m in movies:
        results.append({
            "id": m.id,
            "title": m.title,
            "year": m.year,
            "poster_url": m.poster_url,
            "rating": round(m.average_rating, 1),
            "votes": m.ratings_count,
            "tmdb_id": m.tmdb_id
        })

    return {
        "data": results,
        "total": total,
        "page": page,
        "pages": (total + limit - 1) // limit
    }


# 2. Gatunki
@router.get("/genres", response_model=List[str])
def get_genres(db: Session = Depends(get_db)):
    genres = db.query(Genre.name).order_by(Genre.name).all()
    return [g.name for g in genres]


# 3. Szczegóły
@router.get("/{movie_id}/details")
def get_movie_details(movie_id: int, db: Session = Depends(get_db)):
    movie = db.query(Movie).filter(Movie.id == movie_id).first()
    if not movie:
        raise HTTPException(status_code=404, detail="Film nie znaleziony")

    avg_rating = db.query(func.avg(Rating.rating)).filter(Rating.movie_id == movie_id).scalar() or 0.0

    extras = {"cast": [], "trailer": None, "backdrop": None, "images": []}

    if movie.tmdb_id and TMDB_API_KEY:
        try:
            url = f"https://api.themoviedb.org/3/movie/{movie.tmdb_id}?api_key={TMDB_API_KEY}&append_to_response=credits,videos,images&language=en-US"
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                data = response.json()
                if data.get("backdrop_path"):
                    extras["backdrop"] = f"https://image.tmdb.org/t/p/original{data['backdrop_path']}"
                if "credits" in data:
                    for actor in data["credits"].get("cast", [])[:6]:
                        profile_path = actor.get("profile_path")
                        extras["cast"].append({
                            "name": actor["name"],
                            "character": actor["character"],
                            "photo": f"https://image.tmdb.org/t/p/w200{profile_path}" if profile_path else None
                        })
                if "videos" in data:
                    for video in data["videos"].get("results", []):
                        if video["site"] == "YouTube" and video["type"] == "Trailer":
                            extras["trailer"] = video["key"]
                            break
                if "images" in data:
                    extras["images"] = [
                        f"https://image.tmdb.org/t/p/w500{img['file_path']}"
                        for img in data["images"].get("backdrops", [])[:4]
                    ]
        except Exception as e:
            print(f"Błąd TMDB: {e}")

    return {
        "id": movie.id,
        "title": movie.title,
        "year": movie.year,
        "description": movie.description,
        "poster_url": movie.poster_url,
        "rating": round(float(avg_rating), 1),
        "extras": extras
    }