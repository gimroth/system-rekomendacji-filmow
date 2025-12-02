from datetime import datetime, timedelta
from jose import jwt, JWTError
import os
from dotenv import load_dotenv

load_dotenv()

# Odczyt kluczy i algorytmu z .env
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
# Używamy tej samej stałej dla odświeżania, ale w profesjonalnej aplikacji powinny być inne
REFRESH_SECRET_KEY = SECRET_KEY


def create_access_token(data: dict):
    payload = data.copy()
    # Expire ustawiamy na 30 minut, jak zdefiniowane w .env (ACCESS_TOKEN_EXPIRE_MINUTES)
    expire_minutes = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))
    payload["exp"] = datetime.utcnow() + timedelta(minutes=expire_minutes)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def create_refresh_token(data: dict):
    payload = data.copy()
    # Expire ustawiamy na 7 dni, jak zdefiniowane w .env (REFRESH_TOKEN_EXPIRE_DAYS)
    expire_days = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", 7))
    payload["exp"] = datetime.utcnow() + timedelta(days=expire_days)
    return jwt.encode(payload, REFRESH_SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str):
    """Dekoduje token i zwraca payload lub wywołuje wyjątek JWTError."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError as e:
        # Ten błąd zostanie obsłużony przez dependency injection
        raise e