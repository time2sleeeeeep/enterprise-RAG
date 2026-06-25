# MySQL 客户端：定义 ORM 模型（Document、ChatHistory），初始化数据库表，提供会话依赖注入。

from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from datetime import datetime, timezone

from src.config import settings

engine = create_engine(settings.mysql_url, pool_pre_ping=True, pool_size=10)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id = Column(String(64), primary_key=True)
    filename = Column(String(512), nullable=False)
    file_type = Column(String(32), nullable=False)
    chunk_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    sources = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def init_db():
    """根据 ORM 模型定义在数据库中创建所有表（已存在则跳过）。"""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI 依赖注入生成器：每次请求创建新会话，请求结束后自动关闭。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
