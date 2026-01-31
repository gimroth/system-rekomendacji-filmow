from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func, desc, or_
from typing import List, Optional
import requests
import os
import re


from app.database import get_db
from app.dependencies import get_current_admin
from app.models.user import User
from app.models.movie import Movie
from app.models.rating import Rating
from app.models.comment import Comment
from app.schemas.admin_schemas import (
    MovieAdminResponse,
    TMDBSearchResult,
    MovieStats,
    PaginatedUsersResponse
)

# Konfiguracja klucza TMDB
try:
    from app.config import settings

    TMDB_API_KEY = settings.TMDB_API_KEY
except ImportError:
    TMDB_API_KEY = os.getenv("TMDB_API_KEY")

router = APIRouter(prefix="/admin", tags=["Admin Dashboard"])


# --- 1. UŻYTKOWNICY ---
@router.get("/users", response_model=PaginatedUsersResponse)
def get_all_users(
        skip: int = 0,
        limit: int = 20,
        search: Optional[str] = None,
        db: Session = Depends(get_db),
        admin: User = Depends(get_current_admin)
):
    query = db.query(User)
    if search:
        sf = f"%{search}%"
        query = query.filter(or_(User.username.ilike(sf), User.email.ilike(sf)))

    total = query.count()
    users = query.offset(skip).limit(limit).all()
    return {"total": total, "users": users}


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(user_id: int, db: Session = Depends(get_db), admin: User = Depends(get_current_admin)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user or user.role == 'admin':
        raise HTTPException(status_code=400, detail="Nie można usunąć administratora.")
    db.delete(user)
    db.commit()
    return


# --- 2. FILMY (SZUKANIE TMDB + DODAWANIE + LISTA LOKALNA) ---
@router.get("/movies/search")
def search_movies_in_tmdb(query: str, db: Session = Depends(get_db), admin: User = Depends(get_current_admin)):
    if not query or not TMDB_API_KEY:
        return []

    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={query}&language=en-US"

    try:
        response = requests.get(url, timeout=10)
        tmdb_results = response.json().get("results", [])

        # Pobieramy listę wszystkich tmdb_id, które już mamy w bazie
        existing_tmdb_ids = [m.tmdb_id for m in db.query(Movie.tmdb_id).filter(Movie.tmdb_id.isnot(None)).all()]

        parsed_results = []
        for m in tmdb_results:
            tmdb_id = m["id"]
            # Sprawdzamy, czy ten film już u nas jest
            is_added = tmdb_id in existing_tmdb_ids

            parsed_results.append({
                "tmdb_id": tmdb_id,
                "title": m["title"],
                "year": m.get("release_date", "")[:4],
                "poster_url": f"https://image.tmdb.org/t/p/w200{m.get('poster_path')}" if m.get(
                    "poster_path") else None,
                "is_added": is_added  # Nowe pole informacyjne
            })

        return parsed_results
    except Exception as e:
        print(f"Błąd wyszukiwania TMDB: {e}")
        return []


@router.post("/movies/tmdb/{tmdb_id}", response_model=MovieAdminResponse)
def add_movie_by_tmdb_id(tmdb_id: int, db: Session = Depends(get_db), admin: User = Depends(get_current_admin)):
    if db.query(Movie).filter(Movie.tmdb_id == tmdb_id).first():
        raise HTTPException(400, "Film już istnieje w bazie!")

    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}&language=en-US"
    d = requests.get(url).json()

    new_movie = Movie(
        title=d.get("title"),
        year=int(d.get("release_date", "0")[:4]) if d.get("release_date") else None,
        tmdb_id=tmdb_id,
        description=d.get("overview"),
        poster_url=f"https://image.tmdb.org/t/p/w500{d.get('poster_path')}" if d.get("poster_path") else None
    )
    db.add(new_movie)
    db.commit()
    db.refresh(new_movie)
    return new_movie


@router.get("/movies/local")
def get_local_movies(
        skip: int = 0, limit: int = 50, search: Optional[str] = None,
        db: Session = Depends(get_db), admin: User = Depends(get_current_admin)
):
    query = db.query(Movie)
    if search:
        query = query.filter(Movie.title.ilike(f"%{search}%"))

    total = query.count()

    # Podzapytanie do statystyk
    results = db.query(
        Movie.id, Movie.title, Movie.year, Movie.poster_url, Movie.tmdb_id,
        func.count(Rating.id).label("ratings_count"),
        func.coalesce(func.avg(Rating.rating), 0.0).label("average_rating")
    ).outerjoin(Rating, Movie.id == Rating.movie_id)

    if search:
        results = results.filter(Movie.title.ilike(f"%{search}%"))

    movies = results.group_by(Movie.id).order_by(desc("ratings_count")).offset(skip).limit(limit).all()

    return {
        "total": total,
        "movies": [{
            "id": m.id, "title": m.title, "year": m.year, "poster_url": m.poster_url,
            "ratings_count": m.ratings_count, "average_rating": round(float(m.average_rating), 1)
        } for m in movies]
    }


@router.delete("/movies/{movie_id}")
def delete_movie(movie_id: int, db: Session = Depends(get_db), admin: User = Depends(get_current_admin)):
    m = db.query(Movie).filter(Movie.id == movie_id).first()
    if not m:
        raise HTTPException(404, "Nie znaleziono filmu")
    db.delete(m)
    db.commit()
    return {"status": "ok"}


# --- 3. KOMENTARZE ---
@router.get("/comments")
def get_comments(
        skip: int = 0, limit: int = 50, search: Optional[str] = None,
        db: Session = Depends(get_db), admin: User = Depends(get_current_admin)
):
    query = db.query(Comment).join(User).join(Movie)
    if search:
        sf = f"%{search}%"
        query = query.filter(or_(Comment.content.ilike(sf), User.username.ilike(sf), Movie.title.ilike(sf)))

    total = query.count()
    data = query.order_by(Comment.created_at.desc()).offset(skip).limit(limit).all()
    return {
        "total": total,
        "comments": [{
            "id": c.id, "content": c.content, "username": c.user.username,
            "movie_title": c.movie.title, "movie_poster": c.movie.poster_url
        } for c in data]
    }


@router.delete("/comments/{comment_id}")
def del_comment(comment_id: int, db: Session = Depends(get_db), admin: User = Depends(get_current_admin)):
    c = db.query(Comment).filter(Comment.id == comment_id).first()
    if not c: raise HTTPException(404)
    db.delete(c);
    db.commit();
    return {"status": "ok"}


# --- 4. OCENY ---
@router.get("/ratings")
def get_ratings(
        skip: int = 0, limit: int = 50, search: Optional[str] = None,
        db: Session = Depends(get_db), admin: User = Depends(get_current_admin)
):
    query = db.query(Rating).join(User).join(Movie)
    if search:
        sf = f"%{search}%"
        query = query.filter(or_(User.username.ilike(sf), Movie.title.ilike(sf)))

    total = query.count()
    data = query.order_by(Rating.rated_at.desc()).offset(skip).limit(limit).all()
    return {
        "total": total,
        "ratings": [{
            "id": r.id, "rating": r.rating, "username": r.user.username,
            "movie_title": r.movie.title, "movie_poster": r.movie.poster_url
        } for r in data]
    }


@router.delete("/ratings/{rating_id}")
def del_rating(rating_id: int, db: Session = Depends(get_db), admin: User = Depends(get_current_admin)):
    r = db.query(Rating).filter(Rating.id == rating_id).first()
    if not r: raise HTTPException(404)
    db.delete(r);
    db.commit();
    return {"status": "ok"}


# --- 5. AKTYWNOŚĆ (SZCZEGÓŁY) ---
@router.get("/users/{user_id}/activity")
def get_user_activity(user_id: int, db: Session = Depends(get_db), admin: User = Depends(get_current_admin)):
    # Pobieramy komentarze i oceny z dołączonym modelem filmu, aby mieć dostęp do plakatu
    comments = db.query(Comment).options(joinedload(Comment.movie)).filter(Comment.user_id == user_id).all()
    ratings = db.query(Rating).options(joinedload(Rating.movie)).filter(Rating.user_id == user_id).all()

    return {
        "comments": [{"id": x.id, "content": x.content, "movie": x.movie.title, "poster": x.movie.poster_url} for x in
                     comments],
        "ratings": [{"id": x.id, "score": x.rating, "movie": x.movie.title, "poster": x.movie.poster_url} for x in
                    ratings]
    }


@router.get("/movies/{movie_id}/activity")
def get_movie_activity(movie_id: int, db: Session = Depends(get_db), admin: User = Depends(get_current_admin)):
    # Pobieramy komentarze i oceny z dołączonym modelem użytkownika
    comments = db.query(Comment).options(joinedload(Comment.user)).filter(Comment.movie_id == movie_id).all()
    ratings = db.query(Rating).options(joinedload(Rating.user)).filter(Rating.movie_id == movie_id).all()

    return {
        "comments": [{"id": x.id, "content": x.content, "user": x.user.username} for x in comments],
        "ratings": [{"id": x.id, "score": x.rating, "user": x.user.username} for x in ratings]
    }


# --- 6. ML DASHBOARD ---
@router.get("/ml/metrics")
def get_ml_metrics(admin: User = Depends(get_current_admin)):
    """
    Zwraca metryki modeli ML (ANFIS vs MLP) z pliku comparison_report.txt
    """
    try:
        # Znajdź najnowszy folder comparison
        dirs = [d for d in os.listdir('.') if d.startswith('comparison_anfis_mlp')]

        if not dirs:
            raise FileNotFoundError("Brak folderu comparison")

        latest_dir = max(dirs)  # Najnowszy (sortowanie alfabetyczne)

        # Czytaj comparison_report.txt
        report_path = os.path.join(latest_dir, 'comparison_report.txt')

        if not os.path.exists(report_path):
            raise FileNotFoundError(f"Brak pliku {report_path}")

        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parsuj MAE (gwiazdki)
        mae_match = re.search(r"MAE \(gwiazdki\)\s+([\d.]+)⭐\s+([\d.]+)⭐\s+([\d.]+)⭐", content)
        if not mae_match:
            raise ValueError("Nie znaleziono MAE w raporcie")

        anfis_mae = float(mae_match.group(1))
        mlp_mae = float(mae_match.group(2))
        diff_mae = float(mae_match.group(3))

        # Parsuj RMSE (gwiazdki)
        rmse_match = re.search(r"RMSE \(gwiazdki\)\s+([\d.]+)⭐\s+([\d.]+)⭐\s+([\d.]+)⭐", content)
        if not rmse_match:
            raise ValueError("Nie znaleziono RMSE w raporcie")

        anfis_rmse = float(rmse_match.group(1))
        mlp_rmse = float(rmse_match.group(2))
        diff_rmse = float(rmse_match.group(3))

        # Parsuj improvement %
        improvement_match = re.search(r"MLP jest o ([\d.]+)% dokładniejszy", content)
        improvement_pct = float(improvement_match.group(1)) if improvement_match else 0.0

        # Spróbuj wczytać R2 z comparison_table.csv jeśli dostępne
        r2_anfis = None
        r2_mlp = None
        csv_path = os.path.join(latest_dir, 'comparison_table.csv')
        if os.path.exists(csv_path):
            try:
                import csv
                with open(csv_path, 'r', encoding='utf-8') as cf:
                    reader = csv.reader(cf)
                    for row in reader:
                        if not row: continue
                        key = row[0].strip().lower()
                        if 'r2' in key or 'r^2' in key:
                            # Expect format: R2,anfis,mlp,...
                            try:
                                r2_anfis = float(row[1])
                                r2_mlp = float(row[2])
                            except Exception:
                                pass
                            break
            except Exception:
                pass

        # Parsuj datę
        date_match = re.search(r"Data:\s+([\d-]+\s+[\d:]+)", content)
        comparison_date = date_match.group(1) if date_match else "Unknown"

        # Określ jakość modeli
        def get_quality(mae):
            if mae < 0.3:
                return "EXCELLENT"
            elif mae < 0.4:
                return "GOOD"
            elif mae < 0.5:
                return "AVERAGE"
            else:
                return "POOR"

        # Sprawdź czy model jest załadowany (.pt preferowany, .pkl fallback)
        model_path = None
        pt_path = "app/ml/models/anfis_pytorch_latest.pt"
        pkl_path = "app/ml/models/anfis_latest.pkl"
        if os.path.exists(pt_path):
            model_path = pt_path
        elif os.path.exists(pkl_path):
            model_path = pkl_path

        return {
            "status": "success",
            "anfis": {
                "mae": round(anfis_mae, 2),
                "rmse": round(anfis_rmse, 2),
                "r2": round(r2_anfis, 3) if r2_anfis is not None else None,
                "quality": get_quality(anfis_mae)
            },
            "mlp": {
                "mae": round(mlp_mae, 2),
                "rmse": round(mlp_rmse, 2),
                "r2": round(r2_mlp, 3) if r2_mlp is not None else None,
                "quality": get_quality(mlp_mae)
            },
            "comparison": {
                "winner": "MLP" if mlp_mae < anfis_mae else "ANFIS",
                "improvement_pct": round(improvement_pct, 1),
                "difference_mae": round(diff_mae, 2),
                "difference_rmse": round(diff_rmse, 2),
                "date": comparison_date
            },
            "model": {
                "loaded": bool(model_path),
                "path": model_path or "N/A"
            }
        }

    except FileNotFoundError as e:
        # Brak pliku - zwróć fallback
        return {
            "status": "fallback",
            "error": str(e),
            "anfis": {"mae": 0.35, "rmse": 0.43, "quality": "GOOD"},
            "mlp": {"mae": 0.22, "rmse": 0.29, "quality": "EXCELLENT"},
            "comparison": {
                "winner": "MLP",
                "improvement_pct": 37.0,
                "difference_mae": 0.13,
                "difference_rmse": 0.14,
                "date": "Unknown"
            },
            "model": {"loaded": False, "path": "N/A"}
        }

    except Exception as e:
        # Inny błąd
        return {
            "status": "error",
            "error": str(e),
            "anfis": {"mae": 0.0, "rmse": 0.0, "quality": "N/A"},
            "mlp": {"mae": 0.0, "rmse": 0.0, "quality": "N/A"},
            "comparison": {"winner": "N/A", "improvement_pct": 0.0},
            "model": {"loaded": False}
        }


@router.get("/anfis/explain")
def explain_anfis(
    user_id: Optional[int] = None,
    movie_id: Optional[int] = None,
    match1: Optional[float] = None,
    match2: Optional[float] = None,
    match3: Optional[float] = None,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin)
):
    """Return per-input MF degrees, per-rule firing strengths, normalized weights,
    consequents (if available) and per-rule contributions for a single sample.

    Provide either `user_id`+`movie_id` (uses DataProcessor smart features) or
    direct `match1,match2,match3` values in 0-1.
    """
    from app.ml import loader
    from app.ml.data_processor import DataProcessor
    import numpy as np

    # Build features dict
    features = {}
    dp = DataProcessor(db)
    if user_id is not None and movie_id is not None:
        user_feats, top3, source = dp.get_smart_user_features(user_id)
        movie_feats = dp.get_movie_input_for_anfis(movie_id, top3)
        # map movie_feats (m_story...) to match1..3 according to top3 order
        for i, key in enumerate(top3):
            features[f'match{i+1}'] = float(movie_feats.get(f'm_{key}', 0.5))
    else:
        # use direct values (fallback to 0.5)
        features['match1'] = float(match1) if match1 is not None else 0.5
        features['match2'] = float(match2) if match2 is not None else 0.5
        features['match3'] = float(match3) if match3 is not None else 0.5

    model = loader.get_model()
    raw = loader.get_raw_model()

    if model is None or raw is None:
        raise HTTPException(status_code=404, detail='No ANFIS model loaded')

    # If raw is PyTorchANFIS
    explain = {
        'features': features,
        'model_type': type(raw).__name__,
        'membership': {},
        'rule_labels': None,
        'firing': None,
        'weights': None,
        'consequents': None,
        'per_rule_contribution': None,
        'prediction': None
    }

    try:
        # PyTorch model
        from app.ml.anfis_pytorch import PyTorchANFIS
        if isinstance(raw, PyTorchANFIS):
            # prepare numpy input
            arr = np.array([[features.get('match1', 0.5), features.get('match2', 0.5), features.get('match3', 0.5)]], dtype=np.float32)
            import torch
            raw.eval()
            with torch.no_grad():
                t = torch.from_numpy(arr)
                out, info = raw.forward(t)
                firing = info.get('firing').cpu().numpy().tolist()[0]
                weights = info.get('weights').cpu().numpy().tolist()[0]
                consequents = raw.consequents.detach().cpu().numpy().tolist() if hasattr(raw, 'consequents') else None
                contrib = None
                if consequents is not None:
                    contrib = [w * c for w, c in zip(weights, consequents)]

                explain.update({
                    'firing': firing,
                    'weights': weights,
                    'consequents': consequents,
                    'per_rule_contribution': contrib,
                    'prediction': float(out.cpu().numpy().ravel()[0])
                })

                # membership degrees per input-term
                # compute gaussian mf values using centers/sigmas
                membership = {}
                centers = raw.centers.detach().cpu().numpy()
                sigmas = np.exp(raw.log_sigmas.detach().cpu().numpy())
                for i, mname in enumerate(['match1', 'match2', 'match3']):
                    vals = []
                    x = features.get(mname, 0.5)
                    for j in range(raw.n_mfs):
                        c = centers[i, j]
                        s = sigmas[i, j]
                        deg = float(np.exp(-0.5 * ((x - c) / (s + 1e-8)) ** 2))
                        vals.append({'term': f'mf{j}', 'degree': deg, 'center': float(c), 'sigma': float(s)})
                    membership[mname] = vals

                explain['membership'] = membership
                # create rule labels from combinations
                rule_labels = []
                for comb in raw.rule_combinations:
                    rule_labels.append(tuple([f'mf{idx}' for idx in comb]))
                explain['rule_labels'] = rule_labels

                return explain

    except Exception:
        pass

    # Else assume legacy ANFISRecommender
    try:
        from app.ml.anfis_model import ANFISRecommender
        if isinstance(raw, ANFISRecommender):
            # membership per input-term
            membership = {}
            for name in raw.aspect_names:
                terms = list(raw.input_variables[name].terms)
                vals = []
                for term in terms:
                    deg = float(raw._sample_membership(name, term, features.get(name, 0.5)))
                    vals.append({'term': term, 'degree': deg})
                membership[name] = vals

            # compute firing per rule
            F = raw._compute_firing_matrix([features])  # shape (1, n_rules)
            firing = F[0].tolist()
            s = sum(firing) if sum(firing) > 0 else 1e-8
            weights = (F[0] / s).tolist()
            consequents = raw.consequents.tolist() if raw.consequents is not None else None
            contrib = None
            if consequents is not None:
                contrib = [(w * c) for w, c in zip(weights, consequents)]
            prediction = None
            if consequents is not None:
                prediction = float(np.dot(weights, consequents))
            else:
                # fallback predict
                prediction = float(model.predict(features))

            explain.update({
                'membership': membership,
                'rule_labels': raw.rule_labels,
                'firing': firing,
                'weights': weights,
                'consequents': consequents,
                'per_rule_contribution': contrib,
                'prediction': prediction
            })
            return explain

    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Explain failed: {e}')

    raise HTTPException(status_code=500, detail='Unsupported model type for explainability')