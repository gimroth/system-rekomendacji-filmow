from pydantic import BaseModel, Field
from typing import List

# 1. SCHEMAT WYJŚCIOWY (Dla endpointu GET /preferences/genres)
class GenreSchema(BaseModel):
    """Schemat dla pojedynczego gatunku zwracanego do Frontendu."""
    name: str

    class Config:
        from_attributes = True

# 2. SCHEMAT WYJŚCIOWY (Dla endpointów GET filmów)
class MovieSchema(BaseModel):
    id: int
    title: str
    poster_url: str | None

    class Config:
        from_attributes = True

# 3. SCHEMAT WEJŚCIOWY (Dla kompleksowego endpointu POST /onboarding)
class FinalOnboardingSchema(BaseModel):
    """
    Schemat łączący wszystkie dane z 3 etapów formularza.
    """

    selected_movie_ids: List[int] = Field(
        ...,
        min_length=3,
        max_length=10,
        description="ID filmów wybranych przez użytkownika"
    )

    preferred_genres: List[str] = Field(
        ...,
        min_length=2,
        max_length=10,
        description="Lista ulubionych gatunków (np. 'Action', 'Drama')"
    )

    weight_story: int = Field(..., ge=1, le=5, description="Waga dla Fabuły/Scenariusza (1-5)")
    weight_acting: int = Field(..., ge=1, le=5, description="Waga dla Aktorstwa (1-5)")
    weight_visuals: int = Field(..., ge=1, le=5, description="Waga dla Efektów Wizualnych (1-5)")
    weight_sound: int = Field(..., ge=1, le=5, description="Waga dla Dźwięku/Klimatu (1-5)")
    weight_direction: int = Field(..., ge=1, le=5, description="Waga dla Reżyserii (1-5)")