from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.database import get_db
from app.dependencies import get_current_admin  # Weryfikacja roli admina
from app.models.user import User
from app.schemas.user_schemas import UserResponse

router = APIRouter(prefix="/admin", tags=["Admin (CRUD Users)"])


# --- R: READ (Wszystkich użytkowników) ---
@router.get("/users", response_model=List[UserResponse], status_code=status.HTTP_200_OK)
def get_all_users(db: Session = Depends(get_db), admin: User = Depends(get_current_admin)):
    """Pobiera listę wszystkich użytkowników (wymaga uprawnień admina)."""
    users = db.query(User).all()
    return users


# --- U: UPDATE (Zmiana roli/danych użytkownika) ---
@router.patch("/users/{user_id}/role/{new_role}", response_model=UserResponse)
def update_user_role(user_id: int, new_role: str, db: Session = Depends(get_db),
                     admin: User = Depends(get_current_admin)):
    """Zmienia rolę użytkownika (wymaga uprawnień admina)."""

    if new_role not in ["user", "admin"]:
        raise HTTPException(status_code=400, detail="Nieprawidłowa rola. Dozwolone: 'user' lub 'admin'.")

    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(status_code=404, detail="Użytkownik nie znaleziony")

    # Aktualizacja roli
    user.role = new_role
    db.commit()
    db.refresh(user)

    return user


# --- D: DELETE (Usuwanie użytkownika) ---
@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(user_id: int, db: Session = Depends(get_db), admin: User = Depends(get_current_admin)):
    """Usuwa użytkownika po ID (wymaga uprawnień admina)."""
    user_query = db.query(User).filter(User.id == user_id)
    user = user_query.first()

    if not user:
        raise HTTPException(status_code=404, detail="Użytkownik nie znaleziony")

    user_query.delete(synchronize_session=False)
    db.commit()

    return