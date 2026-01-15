from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.database import get_db
from app.models.rating import Rating
from app.models.user import User
from app.models.movie import Movie  # <--- Musimy zaimportować model Filmu
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
    # 1. Sprawdź, czy użytkownik już ocenił ten film
    existing_rating = db.query(Rating).filter(
        Rating.user_id == current_user.id,
        Rating.movie_id == rating_data.movie_id
    ).first()

    if existing_rating:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Już oceniłeś ten film! Możesz edytować swoją ocenę w panelu użytkownika."
        )

    # 2. Oblicz średnią ocenę użytkownika (z 5 aspektów)
    aspects = [
        rating_data.story, 
        rating_data.acting, 
        rating_data.visuals, 
        rating_data.sound, 
        rating_data.direction
    ]
    # Wyliczamy średnią tego konkretnego użytkownika (np. 4.2)
    user_avg_score = round(sum(aspects) / len(aspects), 1)

    # 3. Zapisz ocenę w tabeli 'ratings'
    new_rating = Rating(
        user_id=current_user.id,
        movie_id=rating_data.movie_id,
        rating=user_avg_score,
        story=rating_data.story,
        acting=rating_data.acting,
        visuals=rating_data.visuals,
        sound=rating_data.sound,
        direction=rating_data.direction
    )
    
    db.add(new_rating)
    db.commit()
    db.refresh(new_rating)

    # --- NOWA CZĘŚĆ: AKTUALIZACJA GŁÓWNEGO RANKINGU FILMU ---
    # 4. Pobierz wszystkie oceny dla tego filmu, żeby wyliczyć nową średnią globalną
    all_ratings = db.query(Rating).filter(Rating.movie_id == rating_data.movie_id).all()
    
    if all_ratings:
        # Sumujemy oceny wszystkich użytkowników
        total_sum = sum([float(r.rating) for r in all_ratings])
        # Dzielimy przez liczbę głosów
        global_average = total_sum / len(all_ratings)
        
        # 5. Znajdź film i zaktualizuj jego kolumnę 'rating'
        movie_to_update = db.query(Movie).filter(Movie.id == rating_data.movie_id).first()
        if movie_to_update:
            movie_to_update.rating = round(global_average, 1)
            db.add(movie_to_update)
            db.commit()
            db.refresh(movie_to_update)
            
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
    
    if rating:
        return {"rated": True, "score": rating.rating}
    return {"rated": False}