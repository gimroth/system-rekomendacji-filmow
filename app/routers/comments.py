from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.database import get_db
from app.models.comment import Comment
from app.models.user import User
from app.schemas.comment_schemas import CommentCreate, CommentResponse, CommentUpdate
from app.dependencies import get_current_user

router = APIRouter(
    prefix="/comments",
    tags=["comments"]
)

# 1. Dodawanie komentarza
@router.post("/", response_model=CommentResponse)
async def create_comment(
    comment: CommentCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    new_comment = Comment(
        content=comment.content,
        movie_id=comment.movie_id,
        user_id=current_user.id
    )
    db.add(new_comment)
    db.commit()
    db.refresh(new_comment)
    
    return CommentResponse(
        id=new_comment.id,
        content=new_comment.content,
        created_at=new_comment.created_at,
        movie_id=new_comment.movie_id,
        username=current_user.username,
        user_id=current_user.id
    )

# 2. Pobieranie komentarzy (publiczne)
@router.get("/movie/{movie_id}", response_model=List[CommentResponse])
async def read_comments(movie_id: int, db: Session = Depends(get_db)):
    comments = db.query(Comment).filter(Comment.movie_id == movie_id).order_by(Comment.created_at.desc()).all()
    
    return [
        CommentResponse(
            id=c.id,
            content=c.content,
            created_at=c.created_at,
            movie_id=c.movie_id,
            username=c.user.username if c.user else "Nieznany",
            user_id=c.user_id
        ) for c in comments
    ]

# 3. Usuwanie komentarza
@router.delete("/{comment_id}")
async def delete_comment(
    comment_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    comment_query = db.query(Comment).filter(Comment.id == comment_id)
    comment = comment_query.first()

    if not comment:
        raise HTTPException(status_code=404, detail="Komentarz nie istnieje")

    # Sprawdzenie uprawnień (czy to autor lub admin)
    if comment.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Nie masz uprawnień do usunięcia tego komentarza")

    comment_query.delete(synchronize_session=False)
    db.commit()
    return {"message": "Usunięto komentarz"}

# 4. Edycja komentarza (NOWE)
@router.put("/{comment_id}", response_model=CommentResponse)
async def update_comment(
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

    # Aktualizacja
    db_comment.content = comment_update.content
    db.commit()
    db.refresh(db_comment)

    return CommentResponse(
        id=db_comment.id,
        content=db_comment.content,
        created_at=db_comment.created_at,
        movie_id=db_comment.movie_id,
        username=current_user.username,
        user_id=current_user.id
    )