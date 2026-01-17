from sqlalchemy import Boolean, Column, Integer, String, DateTime
from sqlalchemy.sql import func
from .database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True, nullable=True)
    hashed_password = Column(String)
    
    # 积分余额 (默认为 10，注册送)
    credit_balance = Column(Integer, default=10)
    
    # 用户自定义 Key (Bring Your Own Key)
    openai_api_key = Column(String, nullable=True)
    openai_base_url = Column(String, nullable=True)
    
    azure_api_key = Column(String, nullable=True)
    mistral_api_key = Column(String, nullable=True)
    
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Task(Base):
    __tablename__ = "tasks"

    id = Column(String, primary_key=True, index=True) # UUID string
    user_id = Column(Integer, index=True) # Foreign Key (logical)
    
    status = Column(String, default="pending") # pending, processing, completed, failed
    progress = Column(Integer, default=0) # 0-100
    
    original_filename = Column(String) # For display
    file_hash = Column(String, index=True) # For deduplication
    
    # Paths (relative to storage root or absolute)
    image_path = Column(String)
    result_xml_path = Column(String, nullable=True)
    
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

