from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from app.database import get_db
from app.dependencies import get_current_user
from app.models.user import UserPreference
from app.models.user_preferred_genre import UserPreferredGenre
from app.models.genre import Genre
from app.schemas.preferences import GenreSchema, FinalOnboardingSchema

router = APIRouter(
    prefix="/preferences",
    tags=["Preferences"]
)

@router.get("/genres", response_model=List[GenreSchema], status_code=status.HTTP_200_OK)
def get_available_genres(db: Session = Depends(get_db)):
    """Pobiera listę wszystkich dostępnych gatunków z bazy danych."""
    genres = db.query(Genre).all()
    return genres

@router.post("/onboarding", status_code=status.HTTP_201_CREATED)
def complete_onboarding(
        prefs: FinalOnboardingSchema,
        db: Session = Depends(get_db),
        current_user = Depends(get_current_user)  
):
    """
    Finalizuje onboarding: zapisuje wagi aspektów i ulubione gatunki.
    """

    # 1. Sprawdź czy użytkownik już przeszedł onboarding
    existing_prefs = db.query(UserPreference).filter(
        UserPreference.user_id == current_user.id
    ).first()

    if existing_prefs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Użytkownik już przeszedł onboarding."
        )

    # 2. Zapisz wagi do user_preferences
    db_prefs = UserPreference(
        user_id=current_user.id,
        weight_story=prefs.weight_story,
        weight_acting=prefs.weight_acting,
        weight_visuals=prefs.weight_visuals,
        weight_sound=prefs.weight_sound,
        weight_direction=prefs.weight_direction,
        onboarding_completed=True
    )
    db.add(db_prefs)

    # 3. Zapisz preferred genres do user_preferred_genres
    for genre_name in prefs.preferred_genres:
        genre = db.query(Genre).filter(Genre.name == genre_name).first()
        if genre:
            user_genre = UserPreferredGenre(
                user_id=current_user.id,
                genre_id=genre.id
            )
            db.add(user_genre)

    # 4. Commit wszystkich zmian
    db.commit()

    return {
        "message": "Preferencje zostały zapisane pomyślnie!",
        "preferred_genres": prefs.preferred_genres
    }