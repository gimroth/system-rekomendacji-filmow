from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session, joinedload
from typing import List
from datetime import datetime  # Ważny import

from app.database import get_db
from app.models.comment import Comment
from app.models.user import User
from app.models.movie import Movie
from app.schemas.comment_schemas import CommentCreate, CommentResponse, CommentUpdate
from app.dependencies import get_current_user

router = APIRouter(
    prefix="/comments",
    tags=["comments"]
)

# 1. Dodawanie komentarza
@router.post("/", response_model=CommentResponse)
def create_comment(
    comment: CommentCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    movie = db.query(Movie).filter(Movie.id == comment.movie_id).first()
    if not movie:
        raise HTTPException(status_code=404, detail="Film nie istnieje")

    # Generujemy czas w Pythonie
    current_time = datetime.now()

    new_comment = Comment(
        content=comment.content,
        movie_id=comment.movie_id,
        user_id=current_user.id,
        created_at=current_time
    )
    
    db.add(new_comment)
    db.commit()
    db.refresh(new_comment)

    return CommentResponse(
        id=new_comment.id,
        content=new_comment.content,
        created_at=current_time,
        movie_id=new_comment.movie_id,
        user_id=new_comment.user_id,
        username=current_user.username
    )

# 2. Pobieranie komentarzy (POPRAWIONE)
@router.get("/movie/{movie_id}", response_model=List[CommentResponse])
def read_comments(movie_id: int, db: Session = Depends(get_db)):
    comments = db.query(Comment)\
        .options(joinedload(Comment.user))\
        .filter(Comment.movie_id == movie_id)\
        .order_by(Comment.created_at.desc())\
        .all()
    
    results = []
    for c in comments:
        # ZABEZPIECZENIE:
        # Jeśli w bazie jest stary komentarz bez daty (None), 
        # podstawiamy "teraz", żeby nie wywaliło błędu 500.
        safe_date = c.created_at if c.created_at is not None else datetime.now()
        
        # ZABEZPIECZENIE 2:
        # Jeśli użytkownik został usunięty z bazy, wpisujemy "Nieznany"
        safe_username = c.user.username if c.user else "Nieznany użytkownik"

        results.append(
            CommentResponse(
                id=c.id,
                content=c.content,
                created_at=safe_date,  # Używamy bezpiecznej daty
                movie_id=c.movie_id,
                user_id=c.user_id,
                username=safe_username
            )
        )
    
    return results

# 3. Usuwanie komentarza
@router.delete("/{comment_id}")
def delete_comment(
    comment_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    comment_query = db.query(Comment).filter(Comment.id == comment_id)
    comment = comment_query.first()

    if not comment:
        raise HTTPException(status_code=404, detail="Komentarz nie istnieje")

    if comment.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Brak uprawnień")

    comment_query.delete(synchronize_session=False)
    db.commit()
    return {"message": "Usunięto komentarz"}

# 4. Edycja komentarza
@router.put("/{comment_id}", response_model=CommentResponse)
def update_comment(
    comment_id: int,
    comment_update: CommentUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    comment_query = db.query(Comment).filter(Comment.id == comment_id)
    db_comment = comment_query.first()

    if not db_comment:
        raise HTTPException(status_code=404, detail="Komentarz nie istnieje")

    if db_comment.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Nie możesz edytować cudzego komentarza")

    db_comment.content = comment_update.content
    db.commit()
    db.refresh(db_comment)

    # Zabezpieczenie daty przy edycji
    safe_date = db_comment.created_at if db_comment.created_at else datetime.now()

    return CommentResponse(
        id=db_comment.id,
        content=db_comment.content,
        created_at=safe_date,
        movie_id=db_comment.movie_id,
        user_id=db_comment.user_id,
        username=current_user.username
    )