from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.rating import Rating
from app.models.user import User
from app.schemas.rating import RatingCreate, RatingResponse
from app.dependencies import get_current_user

router = APIRouter(
    prefix="/ratings",
    tags=["ratings"]
)

@router.post("/", response_model=RatingResponse)
def create_rating(
    rating_data: RatingCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # 1. Sprawdzenie czy ocena już istnieje (dzięki Twojemu UniqueConstraint w bazie
    # moglibyśmy łapać błąd IntegrityError, ale zapytanie jest milsze dla użytkownika)
    existing_rating = db.query(Rating).filter(
        Rating.user_id == current_user.id,
        Rating.movie_id == rating_data.movie_id
    ).first()

    if existing_rating:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Już oceniłeś ten film! Możesz edytować swoją ocenę w panelu użytkownika."
        )

    # 2. Wyliczenie średniej (do kolumny 'rating')
    aspects = [
        rating_data.story, 
        rating_data.acting, 
        rating_data.visuals, 
        rating_data.sound, 
        rating_data.direction
    ]
    # round(x, 1) bo w bazie masz Numeric(2, 1)
    avg_score = round(sum(aspects) / len(aspects), 1)

    # 3. Utworzenie obiektu zgodnego z Twoim modelem
    new_rating = Rating(
        user_id=current_user.id,
        movie_id=rating_data.movie_id,
        rating=avg_score,         # <-- Tu wpisujemy wyliczoną średnią
        story=rating_data.story,
        acting=rating_data.acting,
        visuals=rating_data.visuals,
        sound=rating_data.sound,
        direction=rating_data.direction
        # rated_at uzupełni się samo dzięki server_default=func.now()
    )
    
    db.add(new_rating)
    db.commit()
    db.refresh(new_rating)
    
    return new_rating

@router.get("/user/check/{movie_id}")
def check_user_rating(
    movie_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    rating = db.query(Rating).filter(
        Rating.user_id == current_user.id,
        Rating.movie_id == movie_id
    ).first()
    
    # Zwracamy 'rating' zamiast 'score', żeby pasowało do modelu
    if rating:
        return {"rated": True, "score": rating.rating}
    return {"rated": False}