from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func, desc, or_
from typing import List, Optional
import requests
import os

from app.database import get_db
from app.dependencies import get_current_admin
from app.models.user import User
from app.models.movie import Movie
from app.models.rating import Rating
from app.models.comment import Comment
from app.schemas.admin_schemas import (
    MovieAdminResponse,
    TMDBSearchResult,
    MovieStats,
    PaginatedUsersResponse
)

# Konfiguracja klucza TMDB
try:
    from app.config import settings

    TMDB_API_KEY = settings.TMDB_API_KEY
except ImportError:
    TMDB_API_KEY = os.getenv("TMDB_API_KEY")

router = APIRouter(prefix="/admin", tags=["Admin Dashboard"])


# --- 1. UŻYTKOWNICY ---
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
        sf = f"%{search}%"
        query = query.filter(or_(User.username.ilike(sf), User.email.ilike(sf)))

    total = query.count()
    users = query.offset(skip).limit(limit).all()
    return {"total": total, "users": users}


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(user_id: int, db: Session = Depends(get_db), admin: User = Depends(get_current_admin)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user or user.role == 'admin':
        raise HTTPException(status_code=400, detail="Nie można usunąć administratora.")
    db.delete(user)
    db.commit()
    return


# --- 2. FILMY (SZUKANIE TMDB + DODAWANIE + LISTA LOKALNA) ---
@router.get("/movies/search")
def search_movies_in_tmdb(query: str, db: Session = Depends(get_db), admin: User = Depends(get_current_admin)):
    if not query or not TMDB_API_KEY:
        return []

    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={query}&language=en-US"

    try:
        response = requests.get(url, timeout=10)
        tmdb_results = response.json().get("results", [])

        # Pobieramy listę wszystkich tmdb_id, które już mamy w bazie
        existing_tmdb_ids = [m.tmdb_id for m in db.query(Movie.tmdb_id).filter(Movie.tmdb_id.isnot(None)).all()]

        parsed_results = []
        for m in tmdb_results:
            tmdb_id = m["id"]
            # Sprawdzamy, czy ten film już u nas jest
            is_added = tmdb_id in existing_tmdb_ids

            parsed_results.append({
                "tmdb_id": tmdb_id,
                "title": m["title"],
                "year": m.get("release_date", "")[:4],
                "poster_url": f"https://image.tmdb.org/t/p/w200{m.get('poster_path')}" if m.get(
                    "poster_path") else None,
                "is_added": is_added  # Nowe pole informacyjne
            })

        return parsed_results
    except Exception as e:
        print(f"Błąd wyszukiwania TMDB: {e}")
        return []


@router.post("/movies/tmdb/{tmdb_id}", response_model=MovieAdminResponse)
def add_movie_by_tmdb_id(tmdb_id: int, db: Session = Depends(get_db), admin: User = Depends(get_current_admin)):
    if db.query(Movie).filter(Movie.tmdb_id == tmdb_id).first():
        raise HTTPException(400, "Film już istnieje w bazie!")

    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}&language=en-US"
    d = requests.get(url).json()

    new_movie = Movie(
        title=d.get("title"),
        year=int(d.get("release_date", "0")[:4]) if d.get("release_date") else None,
        tmdb_id=tmdb_id,
        description=d.get("overview"),
        poster_url=f"https://image.tmdb.org/t/p/w500{d.get('poster_path')}" if d.get("poster_path") else None
    )
    db.add(new_movie)
    db.commit()
    db.refresh(new_movie)
    return new_movie


@router.get("/movies/local")
def get_local_movies(
        skip: int = 0, limit: int = 50, search: Optional[str] = None,
        db: Session = Depends(get_db), admin: User = Depends(get_current_admin)
):
    query = db.query(Movie)
    if search:
        query = query.filter(Movie.title.ilike(f"%{search}%"))

    total = query.count()

    # Podzapytanie do statystyk
    results = db.query(
        Movie.id, Movie.title, Movie.year, Movie.poster_url, Movie.tmdb_id,
        func.count(Rating.id).label("ratings_count"),
        func.coalesce(func.avg(Rating.rating), 0.0).label("average_rating")
    ).outerjoin(Rating, Movie.id == Rating.movie_id)

    if search:
        results = results.filter(Movie.title.ilike(f"%{search}%"))

    movies = results.group_by(Movie.id).order_by(desc("ratings_count")).offset(skip).limit(limit).all()

    return {
        "total": total,
        "movies": [{
            "id": m.id, "title": m.title, "year": m.year, "poster_url": m.poster_url,
            "ratings_count": m.ratings_count, "average_rating": round(float(m.average_rating), 1)
        } for m in movies]
    }


@router.delete("/movies/{movie_id}")
def delete_movie(movie_id: int, db: Session = Depends(get_db), admin: User = Depends(get_current_admin)):
    m = db.query(Movie).filter(Movie.id == movie_id).first()
    if not m:
        raise HTTPException(404, "Nie znaleziono filmu")
    db.delete(m)
    db.commit()
    return {"status": "ok"}


# --- 3. KOMENTARZE ---
@router.get("/comments")
def get_comments(
        skip: int = 0, limit: int = 50, search: Optional[str] = None,
        db: Session = Depends(get_db), admin: User = Depends(get_current_admin)
):
    query = db.query(Comment).join(User).join(Movie)
    if search:
        sf = f"%{search}%"
        query = query.filter(or_(Comment.content.ilike(sf), User.username.ilike(sf), Movie.title.ilike(sf)))

    total = query.count()
    data = query.order_by(Comment.created_at.desc()).offset(skip).limit(limit).all()
    return {
        "total": total,
        "comments": [{
            "id": c.id, "content": c.content, "username": c.user.username,
            "movie_title": c.movie.title, "movie_poster": c.movie.poster_url
        } for c in data]
    }


@router.delete("/comments/{comment_id}")
def del_comment(comment_id: int, db: Session = Depends(get_db), admin: User = Depends(get_current_admin)):
    c = db.query(Comment).filter(Comment.id == comment_id).first()
    if not c: raise HTTPException(404)
    db.delete(c);
    db.commit();
    return {"status": "ok"}


# --- 4. OCENY ---
@router.get("/ratings")
def get_ratings(
        skip: int = 0, limit: int = 50, search: Optional[str] = None,
        db: Session = Depends(get_db), admin: User = Depends(get_current_admin)
):
    query = db.query(Rating).join(User).join(Movie)
    if search:
        sf = f"%{search}%"
        query = query.filter(or_(User.username.ilike(sf), Movie.title.ilike(sf)))

    total = query.count()
    data = query.order_by(Rating.rated_at.desc()).offset(skip).limit(limit).all()
    return {
        "total": total,
        "ratings": [{
            "id": r.id, "rating": r.rating, "username": r.user.username,
            "movie_title": r.movie.title, "movie_poster": r.movie.poster_url
        } for r in data]
    }


@router.delete("/ratings/{rating_id}")
def del_rating(rating_id: int, db: Session = Depends(get_db), admin: User = Depends(get_current_admin)):
    r = db.query(Rating).filter(Rating.id == rating_id).first()
    if not r: raise HTTPException(404)
    db.delete(r);
    db.commit();
    return {"status": "ok"}


# --- 5. AKTYWNOŚĆ (SZCZEGÓŁY) ---
@router.get("/users/{user_id}/activity")
def get_user_activity(user_id: int, db: Session = Depends(get_db), admin: User = Depends(get_current_admin)):
    # Pobieramy komentarze i oceny z dołączonym modelem filmu, aby mieć dostęp do plakatu
    comments = db.query(Comment).options(joinedload(Comment.movie)).filter(Comment.user_id == user_id).all()
    ratings = db.query(Rating).options(joinedload(Rating.movie)).filter(Rating.user_id == user_id).all()

    return {
        "comments": [{"id": x.id, "content": x.content, "movie": x.movie.title, "poster": x.movie.poster_url} for x in
                     comments],
        "ratings": [{"id": x.id, "score": x.rating, "movie": x.movie.title, "poster": x.movie.poster_url} for x in
                    ratings]
    }


@router.get("/movies/{movie_id}/activity")
def get_movie_activity(movie_id: int, db: Session = Depends(get_db), admin: User = Depends(get_current_admin)):
    # Pobieramy komentarze i oceny z dołączonym modelem użytkownika
    comments = db.query(Comment).options(joinedload(Comment.user)).filter(Comment.movie_id == movie_id).all()
    ratings = db.query(Rating).options(joinedload(Rating.user)).filter(Rating.movie_id == movie_id).all()

    return {
        "comments": [{"id": x.id, "content": x.content, "user": x.user.username} for x in comments],
        "ratings": [{"id": x.id, "score": x.rating, "user": x.user.username} for x in ratings]
    }