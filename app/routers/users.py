from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.user import User
from app.dependencies import get_current_user
from app.schemas.user_schemas import UserResponse, UserUpdate 
from passlib.context import CryptContext

# Konfiguracja haszowania
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

router = APIRouter(
    prefix="/users",
    tags=["users"]
)

@router.get("/me", response_model=UserResponse)
def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

@router.put("/me", response_model=UserResponse)
def update_user_me(
    user_update: UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    print(f"--- ROZPOCZYNAM EDYCJĘ PROFILU: {current_user.username} ---")
    
    # 1. Pobieramy usera z bazy
    user_db = db.query(User).filter(User.id == current_user.id).first()
    if not user_db:
        raise HTTPException(status_code=404, detail="Użytkownik nie znaleziony")

    # 2. Zmiana Emaila
    if user_update.email and user_update.email != user_db.email:
        existing = db.query(User).filter(User.email == user_update.email).first()
        if existing:
            raise HTTPException(status_code=400, detail="Ten email jest już zajęty")
        
        user_db.email = user_update.email
        db.add(user_db)
        print(f"DEBUG: Zmieniono email na {user_update.email}")

    # 3. Zmiana Hasła (POPRAWIONE NAZWY PÓL)
    if user_update.password:
        # [cite_start]Używamy poprawnej nazwy pola: password_hash (a nie hashed_password) [cite: 74]
        print(f"DEBUG: Próba zmiany hasła. Stary hash: {user_db.password_hash[:10]}...")
        
        new_hash = pwd_context.hash(user_update.password)
        
        # WYMUSZENIE AKTUALIZACJI W BAZIE (bezpośredni SQL update dla pewności)
        # UWAGA: Tutaj była przyczyna błędu -> używamy User.password_hash
        db.query(User).filter(User.id == current_user.id).update(
            {User.password_hash: new_hash}
        )
        
        # Aktualizujemy też obiekt w pamięci, żeby funkcja zwróciła poprawne dane
        user_db.password_hash = new_hash
        print(f"DEBUG: Hasło zaktualizowane. Nowy hash: {new_hash[:10]}...")

    # 4. Zatwierdzenie
    try:
        db.commit()
        db.refresh(user_db)
        print("--- ZMIANY ZATWIERDZONE (COMMIT) ---")
    except Exception as e:
        print(f"BŁĄD ZAPISU DO BAZY: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Błąd zapisu danych")

    return user_db