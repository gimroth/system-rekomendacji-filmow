"""
Insert test ratings for a user and run a quick before/after comparison.
Usage:
  python scripts/insert_and_test_ratings.py --user-id 123 --n 10

This will:
- Compute quick metrics (ANFIS + MLP) on a sampled dataset before insertion
- Insert `n` random ratings for `user_id` to random movies
- Recompute metrics after insertion and print/save a small report
"""
import os, sys, argparse, random
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
import pandas as pd
from datetime import datetime
from app.database import SessionLocal
from app.ml.data_processor import DataProcessor
from app.models.rating import Rating

from app.ml.anfis_model import ANFISRecommender
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def quick_train_eval(df, sample_size=2000, random_state=42):
    # sample
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=random_state)

    # choose features for ANFIS (top-3) like training script
    feature_cols = ['m_story', 'm_acting', 'm_visuals', 'm_sound', 'm_direction']
    correlations = df[feature_cols].corrwith(df['target']).sort_values(ascending=False)
    top_3 = correlations.head(3).index.tolist()

    X = pd.DataFrame()
    for i, feat in enumerate(top_3):
        X[f'match{i+1}'] = df[feat]
    y = df['target']

    # single train/test split for speed
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # ANFIS (hybrid)
    anfis = ANFISRecommender(aspect_names=['match1','match2','match3'])
    anfis.build_fuzzy_system()
    anfis.train(X_train, y_train, ridge_alpha=1e-3)
    anfis_eval = anfis.evaluate(X_test, y_test)

    # MLP
    mlp_features = df[feature_cols]
    X_train_mlp, X_test_mlp, y_train_mlp, y_test_mlp = train_test_split(mlp_features, df['target'], test_size=0.2, random_state=random_state)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_mlp)
    X_test_s = scaler.transform(X_test_mlp)
    mlp = MLPRegressor(hidden_layer_sizes=(64,32,16), max_iter=300, random_state=random_state)
    mlp.fit(X_train_s, y_train_mlp)
    y_pred = np.clip(mlp.predict(X_test_s), 0, 1)
    rmse = float(np.sqrt(mean_squared_error(y_test_mlp, y_pred)))
    mae = float(mean_absolute_error(y_test_mlp, y_pred))
    r2 = float(r2_score(y_test_mlp, y_pred))
    mlp_eval = {'rmse': rmse, 'mae': mae, 'r2': r2}

    return {'anfis': anfis_eval, 'mlp': mlp_eval, 'top3': [t.replace('m_','') for t in top_3]}


def insert_random_ratings(db, user_id, n=10, seed=42):
    rng = random.Random(seed)
    # choose random movies from movies.csv (fallback to IDs 1..1000)
    try:
        movies_df = pd.read_csv('movies.csv')
        movie_ids = movies_df['movieId'].unique().tolist() if 'movieId' in movies_df.columns else movies_df['id'].unique().tolist()
    except Exception:
        movie_ids = list(range(1,1001))

    inserted = []
    for _ in range(n):
        mid = int(rng.choice(movie_ids))
        # random aspects 1-5
        story = rng.randint(1,5)
        acting = rng.randint(1,5)
        visuals = rng.randint(1,5)
        sound = rng.randint(1,5)
        direction = rng.randint(1,5)
        # overall rating: rounded average of aspects (scale 1-5)
        overall = round(((story+acting+visuals+sound+direction)/5.0),1)
        r = Rating(user_id=user_id, movie_id=mid, rating=overall, story=story, acting=acting, visuals=visuals, sound=sound, direction=direction)
        try:
            db.add(r)
            db.commit()
            inserted.append(r)
        except Exception:
            db.rollback()
    return inserted


def upsert_random_ratings(db, user_id, n=10, seed=42):
    """Insert or update ratings: if user/movie pair exists update fields, else insert."""
    rng = random.Random(seed)
    try:
        movies_df = pd.read_csv('movies.csv')
        movie_ids = movies_df['movieId'].unique().tolist() if 'movieId' in movies_df.columns else movies_df['id'].unique().tolist()
    except Exception:
        movie_ids = list(range(1,1001))

    upserted = 0
    for _ in range(n):
        mid = int(rng.choice(movie_ids))
        story = rng.randint(1,5)
        acting = rng.randint(1,5)
        visuals = rng.randint(1,5)
        sound = rng.randint(1,5)
        direction = rng.randint(1,5)
        overall = round(((story+acting+visuals+sound+direction)/5.0),1)

        existing = db.query(Rating).filter(Rating.user_id == user_id, Rating.movie_id == mid).first()
        if existing:
            existing.rating = overall
            existing.story = story
            existing.acting = acting
            existing.visuals = visuals
            existing.sound = sound
            existing.direction = direction
            try:
                db.add(existing)
                db.commit()
                upserted += 1
            except Exception:
                db.rollback()
        else:
            r = Rating(user_id=user_id, movie_id=mid, rating=overall, story=story, acting=acting, visuals=visuals, sound=sound, direction=direction)
            try:
                db.add(r)
                db.commit()
                upserted += 1
            except Exception:
                db.rollback()
    return upserted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--user-id', type=int, required=True)
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--force', action='store_true', help='Upsert existing ratings (insert or update)')
    parser.add_argument('--sample-size', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    db = SessionLocal()
    dp = DataProcessor(db)

    print('Collecting training data...')
    df = dp.get_training_data()
    if df.empty:
        print('No training data in DB - abort')
        return

    print('Running quick eval BEFORE insertion...')
    before = quick_train_eval(df, sample_size=args.sample_size, random_state=args.seed)
    print('BEFORE:', before)

    if args.force:
        print(f'Upserting {args.n} random ratings for user {args.user_id} (force mode)...')
        upserted = upsert_random_ratings(db, args.user_id, n=args.n, seed=args.seed)
        print(f'Upserted {upserted} ratings')
        inserted_count = int(upserted)
    else:
        print(f'Inserting {args.n} random ratings for user {args.user_id}...')
        inserted = insert_random_ratings(db, args.user_id, n=args.n, seed=args.seed)
        print(f'Inserted {len(inserted)} ratings')
        inserted_count = len(inserted)

    # refresh data
    df2 = dp.get_training_data()
    print('Running quick eval AFTER insertion...')
    after = quick_train_eval(df2, sample_size=args.sample_size, random_state=args.seed)
    print('AFTER:', after)

    # summary
    report = {
        'timestamp': datetime.now().isoformat(),
        'user_id': args.user_id,
        'n_inserted': inserted_count,
        'before': before,
        'after': after
    }
    outdir = f"results_online_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(outdir, exist_ok=True)
    import json
    with open(os.path.join(outdir, 'report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print('Report saved to', outdir)
    db.close()

if __name__ == '__main__':
    main()
