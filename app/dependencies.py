from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from jose import JWTError
from app.database import get_db
from app.utils.jwt_handler import decode_token
from app.models.user import User

# Schemat zabezpieczeń OAuth2: używamy "Bearer" token w nagłówku Authorization
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """
    Dependency do weryfikacji tokenu dostępu i pobrania użytkownika.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Błędne poświadczenia lub wygasły token",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        # Dekodowanie tokenu
        payload = decode_token(token)
        user_id: int = payload.get("user_id")
        if user_id is None:
            raise credentials_exception

    except JWTError:
        raise credentials_exception

    # Pobranie użytkownika z bazy danych
    user = db.query(User).filter(User.id == user_id).first()

    if user is None:
        raise credentials_exception

    return user


def get_current_admin(current_user: User = Depends(get_current_user)):
    """
    Dependency do sprawdzania, czy zalogowany użytkownik ma rolę 'admin'.
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Brak uprawnień administratora"
        )
    return current_user