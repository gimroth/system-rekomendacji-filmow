from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_
from typing import Optional

from app.database import get_db
from app.models.movie import Movie
from app.models.rating import Rating
from app.models.genre import Genre, MovieGenre

router = APIRouter(prefix="/homepage", tags=["Homepage"])


@router.get("/latest")
def get_latest_movies(
        limit: int = Query(20, ge=1, le=50),
        user_id: Optional[int] = None,
        db: Session = Depends(get_db)
):
    """
    Najnowsze filmy (sortowanie po roku wydania DESC).
    Jeśli user_id - ukrywa już ocenione.
    """
    query = db.query(Movie).filter(Movie.year.isnot(None))

    # Wykluczenie ocenionych filmów
    if user_id:
        rated_movie_ids = db.query(Rating.movie_id).filter(
            Rating.user_id == user_id
        ).subquery()
        query = query.filter(~Movie.id.in_(rated_movie_ids))

    movies = query.order_by(desc(Movie.year)).limit(limit).all()

    return {
        "category": "Najnowsze Premiery",
        "movies": [
            {
                "id": m.id,
                "title": m.title,
                "year": m.year,
                "poster_url": m.poster_url,
                "genres": [g.name for g in m.genres]
            }
            for m in movies
        ]
    }


@router.get("/top-by-aspect")
def get_top_by_aspect(
        aspect: str = Query(..., regex="^(story|acting|visuals|sound|direction)$"),
        limit: int = Query(20, ge=1, le=50),
        user_id: Optional[int] = None,
        db: Session = Depends(get_db)
):
    """
    Top filmy według konkretnego aspektu.

    aspect: story | acting | visuals | sound | direction
    """
    # Mapowanie nazw aspektów na kolumny
    aspect_map = {
        "story": Rating.story,
        "acting": Rating.acting,
        "visuals": Rating.visuals,
        "sound": Rating.sound,
        "direction": Rating.direction
    }

    aspect_column = aspect_map[aspect]

    # Polskie nazwy kategorii
    category_names = {
        "story": "Najlepsza Fabuła",
        "acting": "Najlepsze Aktorstwo",
        "visuals": "Najlepsze Efekty Wizualne",
        "sound": "Najlepszy Dźwięk/Klimat",
        "direction": "Najlepsza Reżyseria"
    }

    # Query: średnia aspektu per film
    query = db.query(
        Movie,
        func.avg(aspect_column).label('avg_aspect')
    ).join(Rating, Movie.id == Rating.movie_id)

    # Wykluczenie ocenionych filmów
    if user_id:
        rated_movie_ids = db.query(Rating.movie_id).filter(
            Rating.user_id == user_id
        ).subquery()
        query = query.filter(~Movie.id.in_(rated_movie_ids))

    # Grupuj i sortuj
    query = query.group_by(Movie.id).having(
        func.count(Rating.id) >= 3  # Min 3 oceny
    ).order_by(desc('avg_aspect')).limit(limit)

    results = query.all()

    return {
        "category": category_names[aspect],
        "aspect": aspect,
        "movies": [
            {
                "id": m.Movie.id,
                "title": m.Movie.title,
                "year": m.Movie.year,
                "poster_url": m.Movie.poster_url,
                "avg_aspect": round(m.avg_aspect, 1),
                "genres": [g.name for g in m.Movie.genres]
            }
            for m in results
        ]
    }


@router.get("/all-categories")
def get_all_categories(
        user_id: Optional[int] = None,
        limit_per_category: int = Query(20, ge=1, le=50),
        db: Session = Depends(get_db)
):
    """
    Zwraca wszystkie kategorie jednym requestem (optymalizacja).
    """
    categories = []

    # 1. Najnowsze
    latest = get_latest_movies(limit=limit_per_category, user_id=user_id, db=db)
    categories.append(latest)

    # 2-6. Top by aspect
    for aspect in ['story', 'acting', 'visuals', 'sound', 'direction']:
        top = get_top_by_aspect(
            aspect=aspect,
            limit=limit_per_category,
            user_id=user_id,
            db=db
        )
        categories.append(top)

    return {
        "user_id": user_id,
        "categories": categories
    }