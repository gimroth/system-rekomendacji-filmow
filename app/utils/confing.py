import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Ładujemy zmienne środowiskowe z pliku .env
load_dotenv()

class Settings(BaseSettings):

    # 2. Klucz TMDB
    TMDB_API_KEY: str

# Tworzymy instancję ustawień, którą będziemy importować w projekcie
settings = Settings()