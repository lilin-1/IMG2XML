from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta

from db import models
from db.database import get_db
from schemas import UserCreate, Token, UserResponse, UpdateKeysRequest
from auth.auth import authenticate_user, create_access_token, get_password_hash, ACCESS_TOKEN_EXPIRE_MINUTES, get_current_user

router = APIRouter()

def get_user_by_username(db: Session, username: str):
    return db.query(models.User).filter(models.User.username == username).first()

def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

@router.post("/register", response_model=UserResponse)
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = get_user_by_username(db, username=user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="该用户名已被注册")
    
    if user.email:
        db_email = get_user_by_email(db, email=user.email)
        if db_email:
            raise HTTPException(status_code=400, detail="该邮箱已被注册")
    
    hashed_password = get_password_hash(user.password)
    new_user = models.User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password,
        credit_balance=30 # 新用户送30次免费额度
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    # 构造 response，不返回 keys
    return UserResponse(
        username=new_user.username,
        email=new_user.email,
        id=new_user.id,
        is_active=new_user.is_active,
        credit_balance=new_user.credit_balance,
        has_custom_keys=False
    )

@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    # Authenticate
    from auth.auth import verify_password
    user = get_user_by_username(db, username=form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: models.User = Depends(get_current_user)):
    return UserResponse(
        username=current_user.username,
        email=current_user.email,
        id=current_user.id,
        is_active=current_user.is_active,
        credit_balance=current_user.credit_balance,
        has_custom_keys=(current_user.azure_api_key is not None)
    )

@router.put("/keys", response_model=UserResponse)
def update_user_keys(
    keys: UpdateKeysRequest, 
    db: Session = Depends(get_db), 
    current_user: models.User = Depends(get_current_user)
):
    """Update user API keys (BYOK)"""
    # Allow clearing keys by passing empty string? Or just overwriting.
    if keys.openai_api_key is not None:
        current_user.openai_api_key = keys.openai_api_key
    if keys.openai_base_url is not None:
        current_user.openai_base_url = keys.openai_base_url
        
    db.commit()
    db.refresh(current_user)
    
    return UserResponse(
        username=current_user.username,
        email=current_user.email,
        id=current_user.id,
        is_active=current_user.is_active,
        credit_balance=current_user.credit_balance,
        has_custom_keys=(current_user.openai_api_key is not None)
    )
