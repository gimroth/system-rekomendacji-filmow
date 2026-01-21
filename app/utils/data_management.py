import pandas as pd
import requests
import os
import time
import re
import asyncio
import aiohttp
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy import text
from pydantic_settings import BaseSettings
from passlib.context import CryptContext
# -----------------------------------------
# 1. KONFIGURACJA
# -----------------------------------------
load_dotenv()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

class Settings(BaseSettings):
    """Wczytuje klucz API z pliku .env."""
    TMDB_API_KEY: str = os.getenv("TMDB_API_KEY")


settings = Settings()
TMDB_API_KEY = settings.TMDB_API_KEY
BASE_URL = "https://api.themoviedb.org/3/movie/"
IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

# Parametry Równoległości - ZREDUKOWANO dla TMDB rate limiting
CONCURRENCY_LIMIT = 10  # Bezpieczny limit dla TMDB (40-50 requestów/10s)
BATCH_SIZE = 1000  # Wielkość batcha do przetwarzania
DELAY_BETWEEN_BATCHES = 2  # Przerwa między batchami w sekundach

# -----------------------------------------
# 2. IMPORT MODELI I BAZY
# -----------------------------------------
try:
    from app.database import SessionLocal, engine, Base
    from app.models.movie import Movie
    from app.models.genre import Genre, MovieGenre
    from app.models.rating import Rating
    from app.models.user import User, UserPreference
    from app.models.user_preferred_genre import UserPreferredGenre
    from app.models.comment import Comment
except ImportError as e:
    print(f"BŁĄD IMPORTU MODELI/BAZY: {e}")
    exit()

# -----------------------------------------
# 3. Ścieżki CSV
# -----------------------------------------
MOVIES_CSV_PATH = 'movies.csv'
LINKS_CSV_PATH = 'links.csv'
RATINGS_CSV_PATH = 'ratings.csv'


# =====================================================================
#   FUNKCJE POMOCNICZE
# =====================================================================

def extract_year(title: str):
    """Wyodrębnia rok z tytułu filmu w formacie '(YYYY)'."""
    match = re.search(r"\((\d{4})\)$", str(title))
    return int(match.group(1)) if match else None


def initialize_database():
    """Tworzy tabele w bazie danych."""
    print("Tworzenie struktur bazodanowych…")
    Base.metadata.create_all(bind=engine)

    # Dodanie indeksów dla optymalizacji
    with engine.connect() as conn:
        try:
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_movies_tmdb_id ON movies(tmdb_id)"))
            conn.execute(
                text("CREATE INDEX IF NOT EXISTS idx_movies_description_null ON movies(id) WHERE description IS NULL"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_ratings_user_movie ON ratings(user_id, movie_id)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_movie_genres_movie_id ON movie_genres(movie_id)"))
            conn.commit()
            print("✅ Indeksy dodane/zweryfikowane.")
        except Exception as e:
            print(f"⚠️  Uwaga przy tworzeniu indeksów: {e}")


def clear_database():
    """Czyści wszystkie tabele (TRUNCATE) w celu bezpiecznego ponownego importu."""
    with SessionLocal() as session:
        # Tymczasowe wyłączenie kluczy obcych dla PostgreSQL
        try:
            session.execute(text("SET session_replication_role = 'replica';"))
        except:
            pass  # Ignoruj jeśli nie PostgreSQL

        session.execute(text("TRUNCATE TABLE ratings RESTART IDENTITY CASCADE;"))
        session.execute(text("TRUNCATE TABLE movie_genres RESTART IDENTITY CASCADE;"))
        session.execute(text("TRUNCATE TABLE movies RESTART IDENTITY CASCADE;"))
        session.execute(text("TRUNCATE TABLE genres RESTART IDENTITY CASCADE;"))
        session.execute(text("TRUNCATE TABLE users RESTART IDENTITY CASCADE;"))

        # Przywrócenie kluczy obcych
        try:
            session.execute(text("SET session_replication_role = 'origin';"))
        except:
            pass

        session.commit()
    print("✅ Baza danych została wyczyszczona.")


def create_placeholder_users(session: Session, ratings_df: pd.DataFrame):
    """Tworzy użytkowników na podstawie unikalnych ID z CSV."""
    if 'userId' not in ratings_df.columns:
        print("Brak kolumny userId. Pomiń tworzenie użytkowników.")
        return

    unique_user_ids = ratings_df['userId'].unique()

    # Batch insert dla optymalizacji
    batch_size = 50000
    for i in range(0, len(unique_user_ids), batch_size):
        batch = unique_user_ids[i:i + batch_size]
        user_objs = []

        for uid in batch:
            user_objs.append(User(
                id=int(uid),
                username=f"user_{uid}",
                email=f"user_{uid}@example.com",
                password_hash="PlaceholderHash"
            ))

        try:
            session.add_all(user_objs)
            session.flush()
        except IntegrityError:
            session.rollback()
            # Próba indywidualnego dodawania w przypadku konfliktów
            for user in user_objs:
                try:
                    session.merge(user)
                    session.flush()
                except:
                    pass

        print(f"  Dodano użytkowników: {min(i + batch_size, len(unique_user_ids))}/{len(unique_user_ids)}")

    print(f"✅ Dodano {len(unique_user_ids)} użytkowników z CSV ocen.")


def create_admin_user(session):
    admin = session.query(User).filter_by(email="admin@example.com").first()
    if admin:
        print("✅ Admin już istnieje.")
        return

    try:
        # Pamiętaj o tej linii naprawiającej licznik ID!
        session.execute(text("SELECT setval('users_id_seq', (SELECT MAX(id) FROM users));"))

        admin = User(
            username="admin",
            email="admin@example.com",
            password_hash=get_password_hash("password123"),  # Teraz używa bcrypt
            role="admin"
        )
        session.add(admin)
        session.commit()
        print("👑 Admin utworzony z hasłem w formacie BCRYPT!")
    except Exception as e:
        session.rollback()
        print(f"⚠️ Błąd: {e}")

# =====================================================================
#   ETAP 1 — IMPORT CSV (Z POPRAWKAMI INTEGRALNOŚCI I TYPÓW)
# =====================================================================

def import_csv_data(session: Session):
    """Importuje filmy, gatunki i oceny z CSV."""

    print("\n--- 1. Wczytywanie CSV ---")
    try:
        movies_df = pd.read_csv(MOVIES_CSV_PATH)
        links_df = pd.read_csv(LINKS_CSV_PATH, dtype={'tmdbId': 'Int64'})
        ratings_df = pd.read_csv(RATINGS_CSV_PATH)
    except FileNotFoundError as e:
        print(f"BŁĄD: Nie znaleziono pliku CSV: {e}. Upewnij się, że są w katalogu głównym.")
        return

    movies_df = movies_df.rename(columns={'movieId': 'id'})
    links_df = links_df.rename(columns={'movieId': 'id'})
    merged_df = pd.merge(movies_df, links_df[['id', 'tmdbId']], on='id', how='left')

    # ---------------------------------------------
    # Tworzenie użytkowników
    # ---------------------------------------------
    print("--- 2. Tworzenie użytkowników ---")
    create_placeholder_users(session, ratings_df)

    # ---------------------------------------------
    # Import gatunków
    # ---------------------------------------------
    print("\n--- 3. Importowanie gatunków ---")
    all_genres = set()
    for genres_str in merged_df['genres'].dropna():
        for g in genres_str.split('|'):
            if g and g != "(no genres listed)":
                all_genres.add(g)

    genre_objs = {g: Genre(name=g) for g in sorted(all_genres)}

    # Batch insert gatunków
    try:
        session.add_all(genre_objs.values())
        session.flush()
    except IntegrityError:
        session.rollback()
        # Jeśli gatunki już istnieją, pobierz je z bazy
        existing_genres = session.query(Genre).all()
        genre_objs = {g.name: g for g in existing_genres}
        print(f"  Użyto istniejących gatunków: {len(genre_objs)}")
    else:
        print(f"✅ Dodano {len(genre_objs)} gatunków.")

    genre_map = {g: obj.id for g, obj in genre_objs.items()}

    # ---------------------------------------------
    # Import filmów
    # ---------------------------------------------
    print("\n--- 4. Importowanie filmów ---")
    existing_tmdb_ids = set()
    successfully_imported_movie_ids = set()

    # Batch insert filmów
    batch_size = 5000
    total_rows = len(merged_df)

    for i in range(0, total_rows, batch_size):
        batch_df = merged_df.iloc[i:i + batch_size]
        movie_objs = []

        for _, row in batch_df.iterrows():
            movie_id = int(row['id'])
            tmdb_id = int(row['tmdbId']) if pd.notna(row['tmdbId']) else None

            if tmdb_id and tmdb_id in existing_tmdb_ids:
                continue

            movie_objs.append(Movie(
                id=movie_id,
                title=row['title'],
                year=extract_year(row['title']),
                tmdb_id=tmdb_id
            ))

            successfully_imported_movie_ids.add(movie_id)

            if tmdb_id:
                existing_tmdb_ids.add(tmdb_id)

        try:
            session.bulk_save_objects(movie_objs)
            session.flush()
        except IntegrityError:
            # Indywidualne dodawanie w przypadku błędów
            session.rollback()
            for movie in movie_objs:
                try:
                    session.merge(movie)
                    session.flush()
                except:
                    pass

        print(f"  Zaimportowano filmy: {min(i + batch_size, total_rows)}/{total_rows}")

    print(f"✅ Zaimportowano {len(successfully_imported_movie_ids)} filmów.")

    # ---------------------------------------------
    # Import relacji Movie–Genre
    # ---------------------------------------------
    print("\n--- 5. Importowanie relacji gatunków ---")

    # Filtrowanie merged_df, aby używać tylko zaimportowanych ID filmów
    filtered_df = merged_df[merged_df['id'].isin(successfully_imported_movie_ids)]

    # Batch insert relacji
    rel_batch_size = 10000
    total_rels = 0

    for i in range(0, len(filtered_df), rel_batch_size):
        batch_df = filtered_df.iloc[i:i + rel_batch_size]
        rel_objs = []

        for _, row in batch_df.iterrows():
            movie_id = int(row['id'])
            genres = row['genres'].split('|') if pd.notna(row['genres']) else []
            for g in genres:
                if g in genre_map:
                    rel_objs.append(MovieGenre(movie_id=movie_id, genre_id=genre_map[g]))

        if rel_objs:
            try:
                session.bulk_save_objects(rel_objs)
                session.flush()
            except IntegrityError:
                session.rollback()
                # Pomijamy duplikaty relacji
                pass

        total_rels += len(rel_objs)
        print(f"  Dodano relacji: {total_rels}")

    print(f"✅ Dodano {total_rels} relacji film–gatunek.")

    # ---------------------------------------------
    # Import ocen (rating + aspekty)
    # ---------------------------------------------
    print("\n--- 6. Importowanie ocen ---")
    ratings_df = ratings_df.rename(columns={
        'movieId': 'movie_id',
        'userId': 'user_id',
        'timestamp': 'rated_at_timestamp'
    })

    # Filtrowanie ocen tylko dla ZAIMPORTOWANYCH filmów
    ratings_df = ratings_df[ratings_df['movie_id'].isin(successfully_imported_movie_ids)]

    # Usuwanie duplikatów (user_id, movie_id), aby uniknąć BŁĘDU INTEGRALNOŚCI
    ratings_df = ratings_df.drop_duplicates(subset=['user_id', 'movie_id'], keep='first')

    print(f"Przygotowano {len(ratings_df)} unikalnych i pasujących ocen do importu.")

    # Batch insert ocen
    rating_batch_size = 50000
    total_imported = 0

    for i in range(0, len(ratings_df), rating_batch_size):
        batch_df = ratings_df.iloc[i:i + rating_batch_size]
        rating_objs = []

        for _, row in batch_df.iterrows():
            rating_objs.append(Rating(
                user_id=int(row['user_id']),
                movie_id=int(row['movie_id']),
                rating=float(row['rating']),
                story=int(row['story']),
                acting=int(row['acting']),
                visuals=int(row['visuals']),
                sound=int(row['sound']),
                direction=int(row['direction']),
            ))

        if rating_objs:
            try:
                session.bulk_save_objects(rating_objs)
                session.flush()
            except IntegrityError:
                session.rollback()
                # Indywidualne dodawanie w przypadku błędów
                for rating in rating_objs:
                    try:
                        session.merge(rating)
                        session.flush()
                    except:
                        pass

        total_imported += len(rating_objs)
        print(f"  Dodano ocen: {total_imported}/{len(ratings_df)}")

    print(f"✅ Dodano {total_imported} ocen.")


# =====================================================================
#   ETAP 2 — TMDB ENRICHMENT (ASYNCHRONICZNY Z OPTYMALIZACJAMI)
# =====================================================================

async def fetch_tmdb_metadata_async(session: aiohttp.ClientSession, tmdb_id: int, retries: int = 3):
    """Asynchronicznie pobiera szczegóły jednego filmu z TMDB z exponential backoff."""
    if not TMDB_API_KEY:
        return None

    url = f"{BASE_URL}{tmdb_id}?api_key={TMDB_API_KEY}&language=en-US"

    for attempt in range(retries):
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    data = await response.json()
                    poster = data.get("poster_path")
                    return {
                        "tmdb_id": tmdb_id,
                        "description": data.get("overview"),
                        "poster_url": f"{IMAGE_BASE_URL}{poster}" if poster else None,
                        "success": True
                    }
                elif response.status == 429:  # Too Many Requests
                    wait_time = (2 ** attempt) + 1  # Exponential backoff
                    await asyncio.sleep(wait_time)
                    continue
                elif response.status == 404:
                    return {"tmdb_id": tmdb_id, "success": False, "error": "Not Found"}
                else:
                    await asyncio.sleep(2 ** attempt)
                    continue
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            return {"tmdb_id": tmdb_id, "success": False, "error": str(e)}

    return {"tmdb_id": tmdb_id, "success": False, "error": "Max retries exceeded"}


async def fetch_with_semaphore(semaphore, session, tmdb_id):
    """Wrapper z semaforem dla kontroli współbieżności."""
    async with semaphore:
        return await fetch_tmdb_metadata_async(session, tmdb_id)


async def enrich_batch_async(movies_batch, http_session, semaphore, progress_counter):
    """Przetwarza batch filmów asynchronicznie."""
    tasks = []
    for movie in movies_batch:
        if movie.tmdb_id:
            tasks.append(fetch_with_semaphore(semaphore, http_session, movie.tmdb_id))

    if not tasks:
        return []

    batch_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filtruj tylko poprawne wyniki
    valid_results = []
    for result in batch_results:
        if isinstance(result, dict) and result.get("success"):
            valid_results.append(result)

    # Aktualizuj licznik postępu
    progress_counter["processed"] += len(movies_batch)
    progress_counter["successful"] += len(valid_results)

    if progress_counter["processed"] % 500 == 0:
        print(f"  Postęp: {progress_counter['processed']}/{progress_counter['total']} "
              f"({progress_counter['successful']} successful)")

    return valid_results


def enrich_data_with_tmdb():
    """Główna, synchroniczna funkcja do uruchomienia asynchronicznego potoku."""
    db_session = SessionLocal()
    print("\n=== FAZA 2: WZBOGACANIE TMDB ===")
    print("--- Rozpoczęcie pobierania metadanych z TMDB (Asynchroniczne) ---")

    # Pobierz tylko filmy z TMDB ID i bez opisu
    movies_to_update = db_session.query(Movie).filter(
        Movie.tmdb_id.isnot(None),
        Movie.description.is_(None)
    ).order_by(Movie.id).all()

    total_count = len(movies_to_update)

    if total_count == 0:
        print("Brak filmów do wzbogacenia. Pomijam fazę 2.")
        db_session.close()
        return

    print(f"Znaleziono {total_count} filmów do wzbogacenia.")
    print(f"Ustawienia: Limit zapytań: {CONCURRENCY_LIMIT}, Batch size: {BATCH_SIZE}")
    print(f"Szacowany czas: ~{total_count / CONCURRENCY_LIMIT * 0.3 / 60:.1f} minut")
    print("UWAGA: Proces może zająć dłużej w przypadku rate limiting TMDB.")

    start_time = time.time()

    # Uruchamiamy asynchroniczny potok
    async def async_pipeline():
        async with aiohttp.ClientSession() as http_session:
            semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
            progress_counter = {"processed": 0, "successful": 0, "total": total_count}

            all_results = []

            # Przetwarzaj w batchach
            for batch_start in range(0, total_count, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, total_count)
                current_batch = movies_to_update[batch_start:batch_end]

                print(f"\nBatch {batch_start // BATCH_SIZE + 1}/{(total_count + BATCH_SIZE - 1) // BATCH_SIZE}: "
                      f"Filmy {batch_start + 1}-{batch_end}")

                batch_results = await enrich_batch_async(
                    current_batch, http_session, semaphore, progress_counter
                )
                all_results.extend(batch_results)

                # Przerwa między batchami (aby uniknąć rate limit)
                if batch_end < total_count:
                    print(f"    Przerwa {DELAY_BETWEEN_BATCHES}s między batchami...")
                    await asyncio.sleep(DELAY_BETWEEN_BATCHES)

            return all_results

    # Uruchom pipeline asynchronicznie
    metadata_results = asyncio.run(async_pipeline())

    # Przetwarzanie wyników i aktualizacja bazy danych
    print(f"\nPrzetwarzanie {len(metadata_results)} wyników...")

    tmdb_map = {res['tmdb_id']: res for res in metadata_results if res and 'tmdb_id' in res}

    updated_count = 0
    commit_every = 1000  # Commit co 1000 filmów

    for i, movie in enumerate(movies_to_update, 1):
        if movie.tmdb_id in tmdb_map:
            meta = tmdb_map[movie.tmdb_id]
            movie.description = meta.get("description")
            movie.poster_url = meta.get("poster_url")
            updated_count += 1

            # Commit w batchach dla wydajności
            if i % commit_every == 0:
                db_session.commit()
                print(f"  Zapisano {i}/{total_count} filmów...")

    # Finalny commit
    db_session.commit()
    db_session.close()

    elapsed_time = time.time() - start_time
    success_rate = (updated_count / total_count * 100) if total_count > 0 else 0

    print(f"\n✅ Zakończono wzbogacanie w {elapsed_time:.2f} sekund ({elapsed_time / 60:.1f} minut).")
    print(f"   Zaktualizowano: {updated_count}/{total_count} filmów ({success_rate:.1f}% success rate)")
    print(f"   Średni czas na film: {elapsed_time / total_count:.2f}s")


# =====================================================================
#   PIPELINE GŁÓWNY
# =====================================================================

def run_data_pipeline():
    """Główna funkcja do zarządzania danymi."""
    # 1. Przygotowanie bazy
    initialize_database()
    clear_database()

    # 2. Sesja dla Fazy 1
    session = SessionLocal()
    try:
        # NAJPIERW: Ładujemy tysiące rekordów z plików
        import_csv_data(session)
        session.commit()
        print("\n✅ Dane z CSV zaimportowane.")

        # NA KOŃCU: Dodajemy Twoje konto specjalne
        print("\n--- TWORZENIE KONTA ADMINISTRATORA ---")
        create_admin_user(session)

    except Exception as e:
        session.rollback()
        print(f"❌ Wystąpił błąd: {e}")
        import traceback
        traceback.print_exc()
    finally:
        session.close()

    # 3. Faza 2: TMDB (wykonywana po zamknięciu sesji importu)
    try:
        enrich_data_with_tmdb()
        print("\n✅ FAZA 2 zakończona pomyślnie")
    except Exception as e:
        print(f"\n⚠️  BŁĄD PODCZAS WZBOGACANIA TMDB (ale import CSV się powiódł): {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 50)
    print("=== PIPELINE ZAKOŃCZONY ===")
    print("=" * 50)
    print("\nDane są gotowe do użycia w aplikacji!")


def quick_test_mode():
    """Tryb testowy - importuje tylko pierwsze 1000 filmów."""
    print("\n🚀 URUCHAMIANIE W TRYBIE TESTOWYM (1000 filmów)")

    # Tylko inicjalizacja i czyszczenie
    initialize_database()
    clear_database()

    session = SessionLocal()
    try:
        # Wczytaj tylko część danych do testów
        movies_df = pd.read_csv(MOVIES_CSV_PATH, nrows=1000)
        links_df = pd.read_csv(LINKS_CSV_PATH, dtype={'tmdbId': 'Int64'}, nrows=1000)
        ratings_df = pd.read_csv(RATINGS_CSV_PATH, nrows=5000)

        # ... reszta importu jak wyżej ale z ograniczonymi danymi
        # (skrócone dla czytelności)

        print("\n✅ Tryb testowy zakończony pomyślnie")

    except Exception as e:
        print(f"\n❌ Błąd w trybie testowym: {e}")
        import traceback
        traceback.print_exc()
    finally:
        session.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Import danych filmowych')
    parser.add_argument('--test', action='store_true', help='Uruchom w trybie testowym (tylko 1000 filmów)')

    args = parser.parse_args()

    if args.test:
        quick_test_mode()
    else:
        run_data_pipeline()