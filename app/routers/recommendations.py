from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, not_
from typing import List, Optional, Union
import numpy as np
import random

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
    from app.ml.loader import get_model, load_model
    try:
        load_model()
    except Exception:
        pass
    anfis_model = get_model()
    if anfis_model is None:
        print('No ANFIS model loaded')
except Exception as e:
    print(f'Failed to initialize central ANFIS loader: {e}')

# =============================================================================
# Schemas
# =============================================================================
class MovieRecommendation(BaseModel):
    movie_id: int
    title: str
    actual_rating: float  # Prawdziwa średnia ocen z bazy
    match_percentage: int # Procentowe dopasowanie (wynik ANFIS)
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
    # 1. ROZPOZNAWANIE USERA
    real_user_id = None
    if isinstance(user_id, str) and not user_id.isdigit():
        user = db.query(User).filter(User.username == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="Użytkownik nie istnieje")
        real_user_id = user.id
    else:
        real_user_id = int(user_id)

    # 2. PRZYGOTOWANIE DANYCH
    ratings_count = db.query(func.count(Rating.id)).filter(Rating.user_id == real_user_id).scalar()
    processor = DataProcessor(db)
    
    try:
        user_features_all, base_top_3, source = processor.get_smart_user_features(real_user_id)
        preferred_genres = processor.get_smart_user_genres(real_user_id)
    except Exception:
        db.rollback()
        user_features_all, base_top_3, source, preferred_genres = {}, ['story', 'acting', 'visuals'], 'default', []

    # 3. POBIERZ KANDYDATÓW (Zintegrowane obliczanie średniej i filtracja braku ocen)
    try:
        # Najpierw podzapytanie o unikalne ID (Postgres fix)
        id_query = db.query(Movie.id)
        if use_genre_filter and preferred_genres:
            id_query = id_query.join(MovieGenre).join(Genre).filter(Genre.name.in_(preferred_genres))
        
        if exclude_rated:
            rated_ids = [r[0] for r in db.query(Rating.movie_id).filter(Rating.user_id == real_user_id).all()]
            if rated_ids:
                id_query = id_query.filter(not_(Movie.id.in_(rated_ids)))

        subquery = id_query.distinct().subquery()

        # Pobieramy pełne obiekty Movie + obliczamy średnią ocen dynamicznie
        # Kluczowa zmiana: inner join na tabelę Rating lub HAVING count > 0
        candidates_raw = db.query(
            Movie,
            func.avg(Rating.rating).label("average_rating")
        ).join(Rating, Movie.id == Rating.movie_id) \
         .filter(Movie.id.in_(subquery)) \
         .group_by(Movie.id) \
         .having(func.count(Rating.id) > 0) \
         .order_by(func.random()) \
         .limit(200).all()
        
    except Exception as e:
        print(f"❌ Błąd DB: {e}")
        db.rollback()
        candidates_raw = []

    if not candidates_raw:
        return RecommendationsResponse(user_id=real_user_id, user_ratings_count=ratings_count, model_used="anfis",
                                       source=source, preferred_genres=preferred_genres, recommendations=[], message="Brak filmów spełniających kryteria.")

    # 4. SCORING I MAPOWANIE
    predictions = []
    all_potential_keys = [k.replace('u_', '') for k in user_features_all.keys()] if user_features_all else ['story', 'acting', 'visuals']

    for row in candidates_raw:
        movie = row[0]
        avg_rating = row[1]
        try:
            movie_all_stats = processor.get_movie_input_for_anfis(movie.id, all_potential_keys)
            best_movie_feat = max(movie_all_stats, key=movie_all_stats.get).replace('m_', '')
            
            current_top_3 = base_top_3[:2]
            if best_movie_feat not in current_top_3:
                current_top_3.append(best_movie_feat)
            else:
                if len(base_top_3) > 2: current_top_3.append(base_top_3[2])
            
            current_top_3 = current_top_3[:3]
            rating_match = predict_anfis_dynamic_hybrid(real_user_id, movie.id, processor, current_top_3)
            
            if rating_match is not None:
                predictions.append({
                    'movie': movie, 
                    'avg_rating': avg_rating, 
                    'rating_match': rating_match, 
                    'used_features': current_top_3
                })
        except Exception:
            continue

    predictions.sort(key=lambda x: x['rating_match'], reverse=True)
    
    recommendations = []
    for pred in predictions[:limit]:
        movie = pred['movie']
        match_pct = int(pred['rating_match'] * 100)
        
        recommendations.append(MovieRecommendation(
            movie_id=movie.id,
            title=movie.title,
            actual_rating=round(float(pred['avg_rating']), 1),
            match_percentage=min(100, match_pct),
            confidence=get_confidence(pred['rating_match']),
            explanation=generate_explanation_hybrid(movie, pred['used_features']),
            genres=[g.name for g in movie.genres],
            poster_url=movie.poster_url
        ))

    return RecommendationsResponse(user_id=real_user_id, user_ratings_count=ratings_count, model_used="anfis",
                                   source=source, preferred_genres=preferred_genres, recommendations=recommendations, message="Gotowe.")

# =============================================================================
# Helpery
# =============================================================================
def predict_anfis_dynamic_hybrid(user_id, movie_id, processor, top_3_keys):
    try:
        user_input, _ = processor.get_user_input_for_anfis(user_id)
        movie_input = processor.get_movie_input_for_anfis(movie_id, top_3_keys)
        if not user_input or not movie_input: return 0.5
        anfis_input = {}
        for i, key in enumerate(top_3_keys):
            u_val = user_input.get(f'u_{key}', 0.5)
            m_val = movie_input.get(f'm_{key}', 0.5)
            diff = abs(u_val - m_val)
            match_score = np.exp(-25 * (diff ** 2)) 
            anfis_input[f'match{i+1}'] = float(match_score)
        if anfis_model:
            raw_score = float(anfis_model.predict(anfis_input))
            final_score = np.power(raw_score, 0.7) if raw_score > 0.65 else raw_score
            return final_score + random.uniform(0, 0.005)
        return 0.5
    except Exception: return 0.5

def generate_explanation_hybrid(movie, used_features):
    names_pl = {'story': 'fabułę', 'acting': 'aktorstwo', 'visuals': 'wizualia', 'sound': 'dźwięk', 'direction': 'reżyserię'}
    feats = [names_pl.get(f, f) for f in used_features]
    return f"Film pasuje do Twoich preferencji, szczególnie przez {feats[-1]}."

def get_confidence(rating: float) -> str:
    if rating >= 0.85: return "bardzo wysoka"
    elif rating >= 0.70: return "wysoka"
    return "średnia"