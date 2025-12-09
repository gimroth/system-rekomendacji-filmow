from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from app.database import get_db
from app.dependencies import get_current_user
from app.models.user import UserPreference
from app.models.genre import Genre
from app.schemas.preferences import GenreSchema, FinalOnboardingSchema
from app.models.movie import Movie
from app.schemas.preferences import MovieSchema
from app.models.rating import Rating
from sqlalchemy.sql.expression import func # Dodany import func

router = APIRouter(
    prefix="/preferences",
    tags=["Preferences"]
)

# 1. Endpoint GET do pobierania Gatunków
@router.get("/genres", response_model=List[GenreSchema], status_code=status.HTTP_200_OK)
def get_available_genres(db: Session = Depends(get_db)):
    """Pobiera listę wszystkich dostępnych gatunków z bazy danych."""
    genres = db.query(Genre).all()
    return genres

# 2. Endpoint POST do zapisywania preferencji (ZABEZPIECZONY)
@router.post("/onboarding", status_code=status.HTTP_201_CREATED)
def complete_onboarding(
        prefs: FinalOnboardingSchema,
        db: Session = Depends(get_db),
        current_user_id: int = Depends(get_current_user) # Wymaga tokenu JWT
):
    """
    Finalizuje onboarding: zapisuje preferencje użytkownika oraz wirtualne oceny startowe.
    """

    # 1. SPRAWDZENIE, CZY UŻYTKOWNIK JUŻ PRZESZEDŁ ONBOARDING
    existing_prefs = db.query(UserPreference).filter(
        UserPreference.user_id == current_user_id
    ).first()

    if existing_prefs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Użytkownik już przeszedł onboarding. Dane można aktualizować w profilu."
        )

    # 2. ETAP: ZAPISYWANIE WAG I GATUNKÓW (DO TABELI user_preferences)

    user_prefs_data = {
        "user_id": current_user_id,
        "preferred_genres": ", ".join(prefs.preferred_genres), # Konwersja listy na string
        "weight_story": prefs.weight_story,
        "weight_acting": prefs.weight_acting,
        "weight_visuals": prefs.weight_visuals,
        "weight_sound": prefs.weight_sound,
        "weight_direction": prefs.weight_direction,
    }

    db_prefs = UserPreference(**user_prefs_data)
    db.add(db_prefs)

    # 3. ETAP: ZAPISYWANIE WIRTUALNYCH OCEN (DO TABELI ratings)

    new_ratings = []

    for movie_id in prefs.selected_movie_ids:
        # Tworzymy wiersz Rating z maksymalnymi ocenami 5/5/5/5/5
        rating = Rating(
            user_id=current_user_id,
            movie_id=movie_id,
            rating=5.0,
            story=5,
            acting=5,
            visuals=5,
            sound=5,
            direction=5
        )
        new_ratings.append(rating)

    db.add_all(new_ratings)

    # 4. COMMITUJEMY WSZYSTKIE ZMIANY
    db.commit()

    return {
        "message": "Onboarding zakończony pomyślnie. Wirtualne oceny i preferencje zostały zapisane.",
        "ratings_count": len(new_ratings)
    }

# 3. Endpoint GET do wyszukiwania filmów
@router.get("/search_movie", response_model=List[MovieSchema])
def search_movie(
    query: str,
    db: Session = Depends(get_db)
):
    """
    Wyszukuje filmy pasujące do fragmentu tytułu.
    """
    movies = db.query(Movie).filter(
        Movie.title.ilike(f'%{query}%')
    ).limit(10).all()

    return movies

# 4. Endpoint GET do pobierania filmów startowych
@router.get("/onboarding_movies", response_model=List[MovieSchema])
def get_onboarding_movies(db: Session = Depends(get_db)):
    """
    Pobiera 48 filmów do wyświetlenia na ekranie onboardingu.
    """
    movies = db.query(Movie).filter(
        Movie.poster_url.is_not(None)
    ).limit(48).all()

    return movies