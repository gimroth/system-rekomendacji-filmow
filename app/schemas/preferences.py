from pydantic import BaseModel, Field
from typing import List, Optional

# 1. SCHEMAT WYJŚCIOWY (Dla endpointu GET /preferences/genres)
class GenreSchema(BaseModel):
    """Schemat dla pojedynczego gatunku zwracanego do Frontendu."""
    name: str

    class Config:
        from_attributes = True

# 2. SCHEMAT WYJŚCIOWY (Dla endpointów GET filmów)
# Możesz go zostawić, jeśli inne części systemu z niego korzystają
class MovieSchema(BaseModel):
    id: int
    title: str
    poster_url: Optional[str] = None

    class Config:
        from_attributes = True

# 3. SCHEMAT WEJŚCIOWY (Dla Twojego formularza Onboarding)
class FinalOnboardingSchema(BaseModel):
    """
    Schemat łączący dane o gatunkach i wagach.
    Usunięto sekcję selected_movie_ids.
    """

    preferred_genres: List[str] = Field(
        ...,
        min_length=1,  # Zmienione z 2 na 1, jeśli dopuszczasz jeden gatunek
        max_length=10,
        description="Lista ulubionych gatunków"
    )

    # Wagi aspektów (Preferencje użytkownika)
    weight_story: int = Field(..., ge=1, le=5, description="Waga dla Fabuły (1-5)")
    weight_acting: int = Field(..., ge=1, le=5, description="Waga dla Aktorstwa (1-5)")
    weight_visuals: int = Field(..., ge=1, le=5, description="Waga dla Efektów (1-5)")
    weight_sound: int = Field(..., ge=1, le=5, description="Waga dla Dźwięku (1-5)")
    weight_direction: int = Field(..., ge=1, le=5, description="Waga dla Reżyserii (1-5)")