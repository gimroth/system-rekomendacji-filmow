from sqlalchemy import text
from app.database import engine

def fix_comments():
    print("--- NAPRAWA RELACJI KOMENTARZY (PostgreSQL) ---")
    
    # 1. Upewniamy się, że kolumna istnieje
    sql_col = text("ALTER TABLE comments ADD COLUMN IF NOT EXISTS movie_id INTEGER;")
    
    # 2. Naprawiamy klucz obcy (usuwamy stary jeśli błędny i dodajemy nowy)
    sql_fk = text("""
        DO $$ 
        BEGIN 
            IF EXISTS (SELECT 1 FROM information_schema.table_constraints WHERE constraint_name = 'fk_comments_movies') THEN 
                ALTER TABLE comments DROP CONSTRAINT fk_comments_movies; 
            END IF; 
        END $$;
        
        ALTER TABLE comments 
        ADD CONSTRAINT fk_comments_movies 
        FOREIGN KEY (movie_id) 
        REFERENCES movies(id) 
        ON DELETE CASCADE;
    """)

    try:
        with engine.connect() as conn:
            conn.execute(sql_col)
            print("Kolumna movie_id sprawdzona.")
            conn.execute(sql_fk)
            conn.commit()
            print("SUKCES: Relacja komentarzy z filmami została naprawiona w bazie.")
    except Exception as e:
        print(f"Błąd SQL: {e}")

if __name__ == "__main__":
    fix_comments()