from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.database import get_db
from app.models.user import User
from app.utils.hashing import verify_password, hash_password
from app.utils.jwt_handler import create_access_token
import traceback

router = APIRouter(prefix="/auth", tags=["Auth"])

# --- MODELE LOKALNE ---
class Token(BaseModel):
    access_token: str
    token_type: str
    user_role: str # Dodajemy role do odpowiedzi
    user_id: int

class UserLogin(BaseModel):
    username: str
    password: str

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    class Config:
        from_attributes = True

# --- REJESTRACJA ---
@router.post("/register", response_model=UserResponse)
def register(user_data: UserCreate, db: Session = Depends(get_db)):
    try:
        # Sprawdzenie czy user istnieje
        if db.query(User).filter((User.username == user_data.username) | (User.email == user_data.email)).first():
             raise HTTPException(status_code=400, detail="Użytkownik o takim loginie lub emailu już istnieje")

        new_user = User(
            username=user_data.username,
            email=user_data.email,
            password_hash=hash_password(user_data.password),
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return new_user
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=400, detail="Błąd rejestracji")

# --- LOGOWANIE ---
@router.post("/login", response_model=Token)
def login(user_credentials: UserLogin, db: Session = Depends(get_db)):
    print(f"\n--- LOGOWANIE: {user_credentials.username} ---")
    
    try:
        user = db.query(User).filter(User.username == user_credentials.username).first()
        
        if not user:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Nieprawidłowe dane")

        db_pass = getattr(user, 'password_hash', None) or getattr(user, 'password', None)
        
        if not verify_password(user_credentials.password, db_pass):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Nieprawidłowe dane")

        access_token = create_access_token(data={
            "user_id": user.id,
            "sub": user.username,
            "role": user.role 
        })
        
        return {
            "access_token": access_token, 
            "token_type": "bearer",
            "user_role": user.role,
            "user_id": user.id
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")