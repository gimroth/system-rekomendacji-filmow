from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas.user_schemas import UserCreate, UserLogin, UserResponse
from app.models.user import User
from app.utils.hashing import hash_password, verify_password
from app.utils.jwt_handler import create_access_token, create_refresh_token

router = APIRouter(prefix="/auth", tags=["Auth"])

#  Rejestracja
@router.post("/register", response_model=UserResponse)
def register(user_data: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.username == user_data.username).first()
    if existing:
        raise HTTPException(400, "Użytkownik już istnieje")

    new_user = User(
        username=user_data.username,
        email=user_data.email,
        password_hash=hash_password(user_data.password),
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return new_user


#  Logowanie
@router.post("/login")
def login(data: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == data.username).first()
    if not user or not verify_password(data.password, user.password_hash):
        raise HTTPException(401, "Nieprawidłowe dane logowania")

    # DODAJEMY rolę do payloadu tokena, aby get_current_user działało szybciej
    token_data = {"user_id": user.id, "role": user.role}
    access = create_access_token(token_data)
    refresh = create_refresh_token(token_data)

    # Zwracamy rolę i dane użytkownika, aby zapisać je w localStorage
    return {
        "access_token": access,
        "refresh_token": refresh,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "username": user.username,
            "role": user.role
        }
    }
