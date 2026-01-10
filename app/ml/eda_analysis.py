import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import func
from app.database import SessionLocal
from app.models.rating import Rating
from app.models.movie import Movie
from app.models.genre import Genre, MovieGenre

def perform_eda():
    db = SessionLocal()

    # 1. Pobieranie danych o ocenach
    ratings_query = db.query(Rating).all()
    movies_query = db.query(Movie).all()

    if not ratings_query:
        print("Brak ocen w bazie. Dodaj oceny przez system, aby EDA mogło zadziałać!")
        db.close()
        return

    df_ratings = pd.DataFrame([{
        'user_id': r.user_id,
        'rating': float(r.rating),
        'story': r.story,
        'acting': r.acting,
        'visuals': r.visuals,
        'sound': r.sound,
        'direction': r.direction
    } for r in ratings_query])

    print("--- OGÓLNE STATYSTYKI ---")
    print(f"Liczba ocen w bazie: {len(df_ratings)}")
    print(f"Liczba filmów w bazie: {len(movies_query)}")

    # 2. Wykres: Rozkład ocen ogólnych
    plt.figure(figsize=(10, 6))
    sns.countplot(x='rating', data=df_ratings, palette='viridis')
    plt.title('Rozkład ocen ogólnych (Target dla ANFIS)')
    plt.savefig('rating_distribution.png')

    # 3. Wykres: Korelacja aspektów (Kluczowe dla wyboru cech ANFIS)
    plt.figure(figsize=(10, 8))
    # Wybieramy tylko kolumny numeryczne do korelacji
    corr_cols = ['story', 'acting', 'visuals', 'sound', 'direction', 'rating']
    correlation = df_ratings[corr_cols].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Korelacja aspektów (Dobór cech do modelu)')
    plt.savefig('correlation_heatmap.png')

    # 4. Analiza Gatunków (Twoje zadanie: EDA filmów)
    # Pobieramy statystyki gatunków bezpośrednio przez tabelę asocjacyjną
    genre_stats = db.query(Genre.name, func.count(MovieGenre.movie_id).label('count'))\
        .join(MovieGenre, Genre.id == MovieGenre.genre_id)\
        .group_by(Genre.name)\
        .order_by(func.count(MovieGenre.movie_id).desc()).all()

    if genre_stats:
        df_genres = pd.DataFrame(genre_stats, columns=['name', 'count'])
        plt.figure(figsize=(12, 6))
        sns.barplot(x='count', y='name', data=df_genres, palette='magma')
        plt.title('Najpopularniejsze gatunki w bazie danych')
        plt.savefig('genre_analysis.png')

    # 5. Statystyki użytkowników
    user_counts = df_ratings['user_id'].value_counts()
    print("\n--- STATYSTYKI UŻYTKOWNIKÓW ---")
    print(f"Średnia liczba ocen na użytkownika: {user_counts.mean():.2f}")
    print(f"Najbardziej aktywny użytkownik (ID): {user_counts.idxmax()} (ocen: {user_counts.max()})")

    db.close()
    print("\nAnaliza zakończona. Wykresy zapisano w folderze projektu.")

if __name__ == "__main__":
    perform_eda()