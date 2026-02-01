from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, not_, text
from typing import List, Optional, Union
import pickle
import os
import torch
from app.ml.anfis_pytorch import PyTorchANFIS

from app.database import get_db
from app.models import User, Movie, Rating, Genre
from app.models.genre import MovieGenre
from app.ml.data_processor import DataProcessor
from pydantic import BaseModel

router = APIRouter(prefix="/recommendations", tags=["recommendations"])

# =============================================================================
# Wczytaj model ANFIS (central loader)
# =============================================================================

anfis_model = None
try:
    # central loader may have been invoked at app startup; get reference
    load_model = None
    from app.ml.loader import get_model, load_model
    try:
        # ensure loader attempted to load (harmless if already loaded)
        load_model()
    except Exception:
        pass
    anfis_model = get_model()
    if anfis_model is None:
        print('No ANFIS model loaded (central loader returned None)')
    else:
        print('ANFIS model is ready (from central loader)')
except Exception as e:
    print(f'Failed to initialize central ANFIS loader: {e}')

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
    source: str
    preferred_genres: List[str]
    recommendations: List[MovieRecommendation]
    message: Optional[str] = None

# =============================================================================
# Endpoint główny
# =============================================================================

@router.get("/", response_model=RecommendationsResponse)
async def get_recommendations(
    user_id: Union[int, str] = Query(..., description="ID lub Login użytkownika"),
    limit: int = Query(10, ge=1, le=50, description="Liczba rekomendacji"),
    exclude_rated: bool = Query(True, description="Wykluczyć ocenione filmy?"),
    use_genre_filter: bool = Query(True, description="Filtruj po gatunkach?"),
    db: Session = Depends(get_db)
):
    """
    Generuje rekomendacje ANFIS. Odporny na błędy bazy danych (rollback).
    """
    
    # 1. ROZPOZNAWANIE USERA
    real_user_id = None
    if isinstance(user_id, str) and not user_id.isdigit():
        user = db.query(User).filter(User.username == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail=f"Użytkownik o loginie '{user_id}' nie istnieje")
        real_user_id = user.id
    else:
        real_user_id = int(user_id)
        user = db.query(User).filter(User.id == real_user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="Użytkownik nie istnieje")

    # Sprawdzenie modelu
    global anfis_model
    if not anfis_model:
        try:
            from app.ml.loader import get_model, load_model
            load_model()
            anfis_model = get_model()
        except Exception as e:
            print(f"[ERROR] Could not reload ANFIS model: {e}")
            raise HTTPException(status_code=500, detail="Nie można załadować modelu ANFIS.")

    # Policz oceny
    ratings_count = db.query(func.count(Rating.id)).filter(Rating.user_id == real_user_id).scalar()

    # =========================================================================
    # INTELIGENTNE CECHY + GATUNKI (Z ZABEZPIECZENIEM DB)
    # =========================================================================
    processor = DataProcessor(db)
    
    user_features = None
    top_3_keys = ['story', 'acting', 'visuals'] # Fallback
    source = 'default'
    preferred_genres = []

    # Pobieranie cech z bezpiecznym rollbackiem
    try:
        user_features, top_3_keys, source = processor.get_smart_user_features(real_user_id)
    except Exception as e:
        print(f"❌ Błąd DataProcessor (features): {e}")
        # WAŻNE: Resetujemy sesję, aby kolejne zapytania nie padły
        db.rollback()
        source = 'error_fallback'

    # Pobieranie gatunków z bezpiecznym rollbackiem
    try:
        preferred_genres = processor.get_smart_user_genres(real_user_id)
    except Exception as e:
        print(f"❌ Błąd DataProcessor (genres): {e}")
        db.rollback()
        preferred_genres = []

    # =========================================================================
    # POBIERZ KANDYDATÓW
    # =========================================================================
    movies_query = db.query(Movie)

    if exclude_rated:
        try:
            rated_ids = [r[0] for r in db.query(Rating.movie_id).filter(
                Rating.user_id == real_user_id
            ).all()]
            if rated_ids:
                movies_query = movies_query.filter(not_(Movie.id.in_(rated_ids)))
        except Exception as e:
            print(f"❌ Błąd pobierania ocenionych: {e}")
            db.rollback()

    if use_genre_filter and preferred_genres:
        movies_query = movies_query.join(MovieGenre).join(Genre).filter(
            Genre.name.in_(preferred_genres)
        ).distinct()

    try:
        candidates = movies_query.limit(200).all()
    except Exception as e:
        print(f"❌ Błąd pobierania kandydatów: {e}")
        db.rollback()
        candidates = []

    if not candidates:
        return RecommendationsResponse(
            user_id=real_user_id,
            user_ratings_count=ratings_count,
            model_used="anfis",
            source=source,
            preferred_genres=preferred_genres,
            recommendations=[],
            message="Brak filmów (lub błąd bazy danych)."
        )

    # =========================================================================
    # SCORING (ANFIS)
    # =========================================================================
    predictions = []
    for movie in candidates:
        try:
            rating = predict_anfis_dynamic(real_user_id, movie.id, processor, top_3_keys)
            if rating is not None:
                predictions.append({'movie': movie, 'rating': rating})
        except Exception:
            # Tu nie robimy rollback, bo funkcja pomocnicza może nie używać sesji DB w sposób krytyczny
            continue

    predictions.sort(key=lambda x: x['rating'], reverse=True)
    top_predictions = predictions[:limit]

    # =========================================================================
    # RESPONSE
    # =========================================================================
    recommendations = []
    for pred in top_predictions:
        movie = pred['movie']
        rating_denorm = (pred['rating'] * 4) + 1 
        
        movie_genres = [g.name for g in movie.genres] if hasattr(movie, 'genres') else []

        recommendations.append(MovieRecommendation(
            movie_id=movie.id,
            title=movie.title,
            predicted_rating=round(rating_denorm, 2),
            confidence=get_confidence(rating_denorm),
            explanation=generate_explanation_dynamic(real_user_id, movie.id, processor, top_3_keys),
            genres=movie_genres,
            poster_url=getattr(movie, 'poster_url', None)
        ))

    message = f"Źródło: {source}. Top cechy: {', '.join(top_3_keys)}."

    return RecommendationsResponse(
        user_id=real_user_id,
        user_ratings_count=ratings_count,
        model_used="anfis",
        source=source,
        preferred_genres=preferred_genres,
        recommendations=recommendations,
        message=message
    )

# =============================================================================
# Helpery
# =============================================================================

def predict_anfis_dynamic(user_id, movie_id, processor, top_3_keys):
    try:
        user_input, _ = processor.get_user_input_for_anfis(user_id)
        movie_input = processor.get_movie_input_for_anfis(movie_id, top_3_keys)

        if not user_input or not movie_input: return None
        diff = abs(u_val - m_val)

        anfis_input = {}
        for i, key in enumerate(top_3_keys):
            u_val = user_input.get(f'u_{key}', 0.5)
            m_val = movie_input.get(f'm_{key}', 0.5)
            match_score = np.exp(-5 * diff)
            anfis_input[f'match{i+1}'] = match_score

        if anfis_model:
            return anfis_model.predict(anfis_input)
        return 0.5
    except Exception:
        return 0.5

def generate_explanation_dynamic(user_id, movie_id, processor, top_3_keys):
    names_pl = {'story': 'fabułę', 'acting': 'aktorstwo', 'visuals': 'wizualia', 'sound': 'dźwięk', 'direction': 'reżyserię'}
    if not top_3_keys: return "Dopasowano do profilu."
    aspect = top_3_keys[0]
    return f"Film wyróżnia się pod kątem {names_pl.get(aspect, aspect)}."

def get_confidence(rating: float) -> str:
    if rating >= 4.5: return "bardzo wysoka"
    elif rating >= 3.5: return "wysoka"
    return "średnia"