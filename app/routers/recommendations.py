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

    LOGIKA CECH:
    - Priorytet 1: Formularz (jeśli wypełniony)
    - Priorytet 2: Średnie z ocen (jeśli ≥5)
    - Fallback: Domyślne

    LOGIKA GATUNKÓW:
    - Ma formularz + brak ocen → z formularza
    - Brak formularza + ma oceny → z ocen (2-10 najczęstszych)
    - Ma formularz + ma oceny → merge inteligentny
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

    print(f"🔍 User {user_id}:")
    print(f"   Source: {source}")
    print(f"   Top 3: {top_3_keys}")
    print(f"   Genres: {preferred_genres}")

    # =========================================================================
    # POBIERZ KANDYDATÓW
    # =========================================================================
    movies_query = db.query(Movie).filter(Movie.id.isnot(None))

    # Wykluczenie ocenionych
    if exclude_rated:
        rated_ids = [r[0] for r in db.query(Rating.movie_id).filter(
            Rating.user_id == user_id
        ).all()]
        if rated_ids:
            movies_query = movies_query.filter(not_(Movie.id.in_(rated_ids)))

    # FILTROWANIE PO GATUNKACH (jeśli są preferowane)
    if use_genre_filter and preferred_genres:
        print(f"   🎬 Filtrowanie po gatunkach: {preferred_genres}")

        # Filtruj filmy które mają PRZYNAJMNIEJ JEDEN z preferowanych gatunków
        # Używamy join z MovieGenre i Genre
        movies_query = movies_query.join(MovieGenre).join(Genre).filter(
            Genre.name.in_(preferred_genres)
        ).distinct()

    candidates = movies_query.all()

    print(f"   📊 Kandydatów: {len(candidates)}")

    if not candidates:
        return RecommendationsResponse(
            user_id=user_id,
            user_ratings_count=ratings_count,
            model_used="anfis",
            source=source,
            preferred_genres=preferred_genres,
            recommendations=[],
            message="Brak filmów spełniających kryteria. Spróbuj wyłączyć filtr gatunków."
        )

    # =========================================================================
    # GENERUJ PREDYKCJE
    # =========================================================================
    predictions = []

    for movie in candidates:
        try:
            rating = predict_anfis_dynamic(user_id, movie.id, processor, top_3_keys)
            if rating is not None:
                predictions.append({'movie': movie, 'rating': rating})
        except Exception as e:
            print(f"⚠️  Prediction failed for movie {movie.id}: {e}")
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
        rating_norm = pred['rating']
        rating_denorm = (rating_norm * 4) + 1  # 0-1 → 1-5

        recommendations.append(MovieRecommendation(
            movie_id=movie.id,
            title=movie.title,
            predicted_rating=round(rating_denorm, 2),
            confidence=get_confidence(rating_denorm),
            explanation=generate_explanation_dynamic(user_id, movie.id, processor, top_3_keys),
            genres=[mg.genre.name for mg in movie.genres],
            poster_url=getattr(movie, 'poster_url', None)
        ))

    # =========================================================================
    # MESSAGE
    # =========================================================================
    if source == 'form':
        message = f"Rekomendacje ANFIS z formularza ({len(preferred_genres)} gatunków). Top 3: {', '.join(top_3_keys)}."
    elif source == 'ratings':
        message = f"Rekomendacje ANFIS z {ratings_count} ocen ({len(preferred_genres)} gatunków). Top 3: {', '.join(top_3_keys)}."
    else:
        message = f"Rekomendacje ANFIS (domyślne). Top 3: {', '.join(top_3_keys)}."

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
# Helper: Predykcja z dynamicznymi top 3
# =============================================================================

def predict_anfis_dynamic(
    user_id: int,
    movie_id: int,
    processor: DataProcessor,
    top_3_keys: List[str]
) -> Optional[float]:
    """
    Predykcja ANFIS z DYNAMICZNYMI top 3.

    Model przyjmuje: u_aspect1, m_aspect1, u_aspect2, m_aspect2, u_aspect3, m_aspect3
    Endpoint mapuje user top 3 na aspect1/2/3.
    """
    try:
        # Pobierz features użytkownika
        user_input, _ = processor.get_user_input_for_anfis(user_id)

        if not user_input or not top_3_keys:
            return None

        # Pobierz features filmu
        movie_input = processor.get_movie_input_for_anfis(movie_id, top_3_keys)

        if not movie_input:
            return None

        # ✅ KLUCZOWA ZMIANA: Mapuj na GENERYCZNE klucze!
        # DataProcessor zwraca: {'u_story': 0.75, 'm_story': 0.65, ...}
        # Model oczekuje: {'u_aspect1': 0.75, 'm_aspect1': 0.65, ...}

        anfis_input = {
            'u_aspect1': user_input[f'u_{top_3_keys[0]}'],
            'm_aspect1': movie_input[f'm_{top_3_keys[0]}'],
            'u_aspect2': user_input[f'u_{top_3_keys[1]}'],
            'm_aspect2': movie_input[f'm_{top_3_keys[1]}'],
            'u_aspect3': user_input[f'u_{top_3_keys[2]}'],
            'm_aspect3': movie_input[f'm_{top_3_keys[2]}']
        }

        # Predykcja
        prediction = anfis_model.predict(anfis_input)

        return float(np.clip(prediction, 0, 1))

    except Exception as e:
        print(f"❌ ANFIS error: {e}")
        import traceback
        traceback.print_exc()
        return None

# =============================================================================
# Helper: Wyjaśnienie
# =============================================================================

def generate_explanation_dynamic(
    user_id: int,
    movie_id: int,
    processor: DataProcessor,
    top_3_keys: List[str]
) -> str:
    """Generuje wyjaśnienie z dynamicznymi top 3."""
    try:
        user_input, _ = processor.get_user_input_for_anfis(user_id)

        if not user_input or not top_3_keys:
            return "Polecamy na podstawie Twoich preferencji."

        movie_input = processor.get_movie_input_for_anfis(movie_id, top_3_keys)

        if not movie_input:
            return "Polecamy na podstawie Twoich preferencji."

        # Top aspekt (aspect1 = top_3_keys[0])
        top_aspect = top_3_keys[0]
        user_val = user_input[f'u_{top_aspect}']
        movie_val = movie_input[f'm_{top_aspect}']

        # Denormalizuj
        user_rating = (user_val * 4) + 1
        movie_rating = (movie_val * 4) + 1

        # Polskie nazwy
        names_pl = {
            'story': 'fabułę', 'acting': 'aktorstwo',
            'visuals': 'efekty wizualne', 'sound': 'dźwięk',
            'direction': 'reżyserię'
        }
        aspect_pl = names_pl.get(top_aspect, top_aspect)

        # Generuj tekst
        if user_rating >= 4.0 and movie_rating >= 4.0:
            return f"Film ma świetną {aspect_pl} ({movie_rating:.1f}⭐), którą cenisz ({user_rating:.1f}⭐)."
        elif user_rating >= 3.5 and movie_rating >= 3.5:
            return f"Film ma dobrą {aspect_pl} ({movie_rating:.1f}⭐), co pasuje do preferencji."
        else:
            return f"Film ma {aspect_pl} na poziomie {movie_rating:.1f}⭐."

    except Exception as e:
        return "Polecamy na podstawie Twoich preferencji."

def get_confidence(rating: float) -> str:
    """Zwraca poziom pewności."""
    if rating >= 4.5: return "bardzo wysoka"
    elif rating >= 4.0: return "wysoka"
    elif rating >= 3.5: return "średnia"
    elif rating >= 3.0: return "niska"
    else: return "bardzo niska"

# =============================================================================
# Debug endpoint
# =============================================================================

@router.get("/test/{user_id}/{movie_id}")
async def test_prediction(
    user_id: int,
    movie_id: int,
    db: Session = Depends(get_db)
):
    """Test predykcji z pełnymi informacjami o użytkowniku."""

    user = db.query(User).filter(User.id == user_id).first()
    movie = db.query(Movie).filter(Movie.id == movie_id).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")
    if not anfis_model:
        raise HTTPException(status_code=503, detail="ANFIS not loaded")

    processor = DataProcessor(db)
    ratings_count = db.query(func.count(Rating.id)).filter(
        Rating.user_id == user_id
    ).scalar()

    # Pobierz inteligentne cechy + gatunki
    user_features, top_3_keys, source = processor.get_smart_user_features(user_id)
    preferred_genres = processor.get_smart_user_genres(user_id)

    # Predykcja
    prediction = predict_anfis_dynamic(user_id, movie_id, processor, top_3_keys)

    if prediction is None:
        raise HTTPException(status_code=500, detail="Prediction failed")

    prediction_denorm = (prediction * 4) + 1

    return {
        "user_id": user_id,
        "movie_id": movie_id,
        "movie_title": movie.title,
        "model_used": "anfis_dynamic",
        "user_ratings_count": ratings_count,
        "source": source,  # 'form' | 'ratings' | 'default'
        "user_top_3_aspects": top_3_keys,  # Dynamiczne top 3 dla tego usera
        "user_preferred_genres": preferred_genres,  # Gatunki usera
        "predicted_rating_normalized": round(prediction, 4),
        "predicted_rating": round(prediction_denorm, 2),
        "confidence": get_confidence(prediction_denorm),
        "explanation": generate_explanation_dynamic(user_id, movie_id, processor, top_3_keys)
    }

@router.post("/reload-models")
async def reload_models():
    """Przeładuj model ANFIS."""
    load_models()
    return {
        "message": "Model reloaded",
        "status": {"anfis": "loaded" if anfis_model else "not loaded"}
    }