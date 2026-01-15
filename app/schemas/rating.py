from pydantic import BaseModel, Field
from datetime import datetime

class RatingCreate(BaseModel):
    movie_id: int
    # Walidacja wejścia: musi być int 1-5
    story: int = Field(..., ge=1, le=5)
    acting: int = Field(..., ge=1, le=5)
    visuals: int = Field(..., ge=1, le=5)
    sound: int = Field(..., ge=1, le=5)
    direction: int = Field(..., ge=1, le=5)

class RatingResponse(RatingCreate):
    id: int
    user_id: int
    rating: float      # W modelu nazywa się to 'rating' (Numeric)
    rated_at: datetime # W modelu nazywa się to 'rated_at'

    class Config:
        from_attributes = True