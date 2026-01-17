from typing import Optional
from pydantic import BaseModel

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str | None = None

class UserBase(BaseModel):
    username: str
    email: str | None = None

class UserCreate(UserBase):
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(UserBase):
    id: int
    is_active: bool
    credit_balance: int
    has_custom_keys: bool # 为了安全，不返回具体 Key，只返回是否有

    class Config:
        from_attributes = True

class UpdateKeysRequest(BaseModel):
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None

