"""
Router rekomendacji ANFIS z dynamicznymi top 3 + inteligentne gatunki
Autorzy: Emilia (ML integration)
Data: 2026-01-13
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, not_, or_
from typing import List, Optional
import pickle
import numpy as np
import os

from app.database import get_db
from app.models import User, Movie, Rating, Genre
from app.models.genre import MovieGenre
from app.ml.data_processor import DataProcessor
from pydantic import BaseModel

router = APIRouter(prefix="/recommendations", tags=["recommendations"])

# =============================================================================
# Wczytaj model ANFIS
# =============================================================================

ANFIS_MODEL_PATH = "app/ml/models/anfis_latest.pkl"
anfis_model = None

def load_models():
    global anfis_model
    if os.path.exists(ANFIS_MODEL_PATH):
        try:
            with open(ANFIS_MODEL_PATH, 'rb') as f:
                anfis_model = pickle.load(f)
            print(f"✅ ANFIS loaded from {ANFIS_MODEL_PATH}")
        except Exception as e:
            print(f"⚠️  Failed to load ANFIS: {e}")
    else:
        print(f"⚠️  ANFIS not found at {ANFIS_MODEL_PATH}")

load_models()

# =============================================================================
# Schemas
# =============================================================================

class MovieRecommendation(BaseModel):
    movie_id: int
    title: str
    predicted_rating: float
    confidence: str
    explanation: str
    genres: List[str]
    poster_url: Optional[str] = None

    class Config:
        from_attributes = True

class RecommendationsResponse(BaseModel):
    user_id: int
    user_ratings_count: int
    model_used: str
    source: str  # 'form' | 'ratings' | 'default'
    preferred_genres: List[str]  # Gatunki użytkownika
    recommendations: List[MovieRecommendation]
    message: Optional[str] = None

# =============================================================================
# Endpoint główny
# =============================================================================

@router.get("/", response_model=RecommendationsResponse)
async def get_recommendations(
    user_id: int = Query(..., description="ID użytkownika"),
    limit: int = Query(10, ge=1, le=50, description="Liczba rekomendacji"),
    exclude_rated: bool = Query(True, description="Wykluczyć ocenione filmy?"),
    use_genre_filter: bool = Query(True, description="Filtruj po gatunkach?"),
    db: Session = Depends(get_db)
):
    """
    Generuje rekomendacje ANFIS z dynamicznymi top 3 + inteligentne gatunki.
    """

    # Sprawdź user
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not anfis_model:
        raise HTTPException(status_code=503, detail="ANFIS not loaded")

    # Policz oceny
    ratings_count = db.query(func.count(Rating.id)).filter(
        Rating.user_id == user_id
    ).scalar()

    # =========================================================================
    # INTELIGENTNE CECHY + GATUNKI
    # =========================================================================
    processor = DataProcessor(db)

    # Pobierz cechy (dynamiczne top 3)
    user_features, top_3_keys, source = processor.get_smart_user_features(user_id)

    # Pobierz preferowane gatunki
    preferred_genres = processor.get_smart_user_genres(user_id)

    # =========================================================================
    # POBIERZ KANDYDATÓW
    # =========================================================================
    movies_query = db.query(Movie)

    # Wykluczenie ocenionych
    if exclude_rated:
        rated_ids = [r[0] for r in db.query(Rating.movie_id).filter(
            Rating.user_id == user_id
        ).all()]
        if rated_ids:
            movies_query = movies_query.filter(not_(Movie.id.in_(rated_ids)))

    # FILTROWANIE PO GATUNKACH
    if use_genre_filter and preferred_genres:
        movies_query = movies_query.join(MovieGenre).join(Genre).filter(
            Genre.name.in_(preferred_genres)
        ).distinct()

    candidates = movies_query.limit(200).all() # Ograniczenie dla wydajności scoringu

    if not candidates:
        return RecommendationsResponse(
            user_id=user_id,
            user_ratings_count=ratings_count,
            model_used="anfis",
            source=source,
            preferred_genres=preferred_genres,
            recommendations=[],
            message="Brak filmów spełniających kryteria."
        )

    # =========================================================================
    # GENERUJ PREDYKCJE (SCORING)
    # =========================================================================
    predictions = []

    for movie in candidates:
        try:
            rating = predict_anfis_dynamic(user_id, movie.id, processor, top_3_keys)
            if rating is not None:
                predictions.append({'movie': movie, 'rating': rating})
        except Exception:
            continue

    # Sortuj i weź top N
    predictions.sort(key=lambda x: x['rating'], reverse=True)
    top_predictions = predictions[:limit]

    # =========================================================================
    # PRZYGOTUJ RESPONSE
    # =========================================================================
    recommendations = []
    for pred in top_predictions:
        movie = pred['movie']
        rating_denorm = (pred['rating'] * 4) + 1  # Skala 0-1 -> 1-5

        # NAPRAWA BŁĘDU: Pobieranie nazw gatunków bezpośrednio z relacji
        movie_genres = [g.name for g in movie.genres] if hasattr(movie, 'genres') else []

        recommendations.append(MovieRecommendation(
            movie_id=movie.id,
            title=movie.title,
            predicted_rating=round(rating_denorm, 2),
            confidence=get_confidence(rating_denorm),
            explanation=generate_explanation_dynamic(user_id, movie.id, processor, top_3_keys),
            genres=movie_genres,
            poster_url=getattr(movie, 'poster_url', None)
        ))

    # MESSAGE
    if source == 'form':
        message = f"Rekomendacje z formularza. Wybrane cechy: {', '.join(top_3_keys)}."
    elif source == 'ratings':
        message = f"Rekomendacje z Twoich {ratings_count} ocen. Top cechy: {', '.join(top_3_keys)}."
    else:
        message = "Rekomendacje domyślne."

    return RecommendationsResponse(
        user_id=user_id,
        user_ratings_count=ratings_count,
        model_used="anfis",
        source=source,
        preferred_genres=preferred_genres,
        recommendations=recommendations,
        message=message
    )

# =============================================================================
# Helper: Predykcja
# =============================================================================

def predict_anfis_dynamic(user_id, movie_id, processor, top_3_keys):
    user_input, _ = processor.get_user_input_for_anfis(user_id)
    movie_input = processor.get_movie_input_for_anfis(movie_id, top_3_keys)

    if not user_input or not movie_input: return None

    # Obliczamy dopasowanie: 1.0 - abs(film - user)
    # Jeśli film ma 0.8 a user chce 0.8 -> wynik 1.0 (idealnie)
    # Jeśli film ma 0.2 a user chce 0.8 -> wynik 0.4 (słabo)
    anfis_input = {}
    for i, key in enumerate(top_3_keys):
        u_val = user_input[f'u_{key}']
        m_val = movie_input[f'm_{key}']
        match_score = 1.0 - abs(u_val - m_val)
        anfis_input[f'match{i+1}'] = match_score

    return anfis_model.predict(anfis_input)
# =============================================================================
# Helpery pomocnicze
# =============================================================================

def generate_explanation_dynamic(user_id, movie_id, processor, top_3_keys):
    """Wyjaśnienie 'dlaczego polecamy'."""
    names_pl = {'story': 'fabułę', 'acting': 'aktorstwo', 'visuals': 'wizualia', 'sound': 'dźwięk', 'direction': 'reżyserię'}
    aspect = top_3_keys[0]
    return f"Film posiada wysoko ocenianą {names_pl.get(aspect, aspect)}, co pasuje do Twojego profilu."

def get_confidence(rating: float) -> str:
    if rating >= 4.5: return "bardzo wysoka"
    elif rating >= 3.5: return "wysoka"
    return "średnia"

@router.post("/reload-models")
async def reload_models():
    load_models()
    return {"message": "Model reloaded", "status": "ok" if anfis_model else "error"}