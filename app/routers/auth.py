from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.database import get_db
from app.models.user import User
from app.utils.hashing import verify_password
from app.utils.jwt_handler import create_access_token
import traceback

router = APIRouter(prefix="/auth", tags=["Auth"])

# --- MODELE LOKALNE (Żeby na 100% działało) ---
class Token(BaseModel):
    access_token: str
    token_type: str

class UserLogin(BaseModel):
    username: str  # <--- TU BYŁ PROBLEM (Frontend wysyła username, nie email)
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

# --- REJESTRACJA (Uproszczona) ---
from app.utils.hashing import hash_password # Import haszowania

@router.post("/register", response_model=UserResponse)
def register(user_data: UserCreate, db: Session = Depends(get_db)):
    try:
        new_user = User(
            username=user_data.username,
            email=user_data.email,
            password_hash=hash_password(user_data.password),
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return new_user
    except Exception:
        raise HTTPException(status_code=400, detail="Użytkownik już istnieje lub błąd danych")

# --- LOGOWANIE (NAPRAWIONE) ---
@router.post("/login", response_model=Token)
def login(user_credentials: UserLogin, db: Session = Depends(get_db)):
    print(f"\n--- LOGOWANIE START: {user_credentials.username} ---") # Teraz używamy username
    
    try:
        # 1. Szukamy po nazwie użytkownika (username), nie po emailu
        user = db.query(User).filter(User.username == user_credentials.username).first()
        
        if not user:
            print("!!! BŁĄD: Nie znaleziono użytkownika.")
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid Credentials")

        # 2. Wyciągamy hasło z bazy (bezpiecznie)
        db_pass = getattr(user, 'password_hash', None) or getattr(user, 'password', None)
        
        if not db_pass:
             print("!!! BŁĄD KRYTYCZNY: Brak pola hasła w bazie")
             raise HTTPException(status_code=500, detail="Server Error")

        # 3. Weryfikacja hasła
        if not verify_password(user_credentials.password, db_pass):
            print("!!! BŁĄD: Złe hasło.")
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid Credentials")

        # 4. Tworzenie tokena
        access_token = create_access_token(data={
            "user_id": user.id,
            "sub": user.username 
        })
        
        print("DEBUG: Sukces! Token wysłany.")
        return {"access_token": access_token, "token_type": "bearer"}

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"!!! CRASH: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")