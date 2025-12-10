from pydantic import BaseModel
from typing import List, Optional

class MovieBase(BaseModel):
    title: str
    year: Optional[int] = None
    tmdb_id: Optional[int] = None
    description: Optional[str] = None
    poster_url: Optional[str] = None

class MovieCreate(MovieBase):
    pass

class MovieAdminResponse(MovieBase):
    id: int
    class Config:
        from_attributes = True

class TMDBSearchResult(BaseModel):
    tmdb_id: int
    title: str
    year: Optional[str] = None
    poster_url: Optional[str] = None
    overview: Optional[str] = None

class MovieStats(BaseModel):
    id: int
    title: str
    year: Optional[int] = None
    tmdb_id: Optional[int] = None
    ratings_count: int
    average_rating: float
    class Config:
        from_attributes = True

class UserListSchema(BaseModel):
    id: int
    username: str
    email: str
    role: str

    class Config:
        from_attributes = True

class PaginatedUsersResponse(BaseModel):
    total: int
    users: List[UserListSchema]