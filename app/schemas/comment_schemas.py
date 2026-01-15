from pydantic import BaseModel
from datetime import datetime

# To wysyła frontend (użytkownik) przy tworzeniu
class CommentCreate(BaseModel):
    content: str
    movie_id: int

# To wysyła frontend przy edycji (tylko treść)
class CommentUpdate(BaseModel):
    content: str

# To odsyła backend (odpowiedź)
class CommentResponse(BaseModel):
    id: int
    content: str
    created_at: datetime
    username: str
    user_id: int      # <-- DODANE: Ważne do identyfikacji autora
    movie_id: int

    class Config:
        from_attributes = True