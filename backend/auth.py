import hashlib
import secrets
import jwt
import datetime
from typing import Dict, Optional, List
from dataclasses import dataclass
import sqlite3
import logging
from contextlib import contextmanager
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class User:
    id: str
    username: str
    email: str
    role: str
    created_at: str
    last_login: str

class UserManager:
    """Manage users and authentication"""
    
    def __init__(self, db_path: str = "data/users.db"):
        self.db_path = db_path
        self.secret_key = os.getenv("JWT_SECRET", "rft-default-secret-key")
        self.init_database()
    
    def init_database(self):
        """Initialize user database"""
        with self._get_db_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT DEFAULT 'user',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS user_projects (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    project_name TEXT NOT NULL,
                    project_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
    
    @contextmanager
    def _get_db_connection(self):
        """Get database connection with context manager"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def hash_password(self, password: str) -> str:
        """Hash password with salt"""
        salt = secrets.token_hex(16)
        return hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        ).hex() + ':' + salt
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        try:
            stored_hash, salt = password_hash.split(':')
            computed_hash = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt.encode('utf-8'),
                100000
            ).hex()
            return secrets.compare_digest(computed_hash, stored_hash)
        except Exception:
            return False
    
    def create_user(self, username: str, email: str, password: str, role: str = "user") -> Optional[User]:
        """Create new user"""
        try:
            user_id = secrets.token_urlsafe(16)
            password_hash = self.hash_password(password)
            
            with self._get_db_connection() as conn:
                conn.execute(
                    "INSERT INTO users (id, username, email, password_hash, role) VALUES (?, ?, ?, ?, ?)",
                    (user_id, username, email, password_hash, role)
                )
            
            logger.info(f"User created: {username}")
            return self.get_user_by_username(username)
        except sqlite3.IntegrityError:
            logger.error(f"User already exists: {username}")
            return None
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user and update last login"""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM users WHERE username = ?", (username,)
                )
                user_data = cursor.fetchone()
                
                if user_data and self.verify_password(password, user_data['password_hash']):
                    # Update last login
                    conn.execute(
                        "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?",
                        (user_data['id'],)
                    )
                    
                    return User(
                        id=user_data['id'],
                        username=user_data['username'],
                        email=user_data['email'],
                        role=user_data['role'],
                        created_at=user_data['created_at'],
                        last_login=user_data['last_login']
                    )
            return None
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return None
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM users WHERE username = ?", (username,)
                )
                user_data = cursor.fetchone()
                
                if user_data:
                    return User(
                        id=user_data['id'],
                        username=user_data['username'],
                        email=user_data['email'],
                        role=user_data['role'],
                        created_at=user_data['created_at'],
                        last_login=user_data['last_login']
                    )
            return None
        except Exception as e:
            logger.error(f"Failed to get user: {e}")
            return None
    
    def create_session(self, user_id: str, expires_hours: int = 24) -> str:
        """Create user session"""
        session_id = secrets.token_urlsafe(32)
        expires_at = datetime.datetime.now() + datetime.timedelta(hours=expires_hours)
        
        with self._get_db_connection() as conn:
            conn.execute(
                "INSERT INTO user_sessions (session_id, user_id, expires_at) VALUES (?, ?, ?)",
                (session_id, user_id, expires_at)
            )
        
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[User]:
        """Validate user session"""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.execute(
                    "SELECT u.* FROM users u JOIN user_sessions s ON u.id = s.user_id WHERE s.session_id = ? AND s.expires_at > CURRENT_TIMESTAMP",
                    (session_id,)
                )
                user_data = cursor.fetchone()
                
                if user_data:
                    return User(
                        id=user_data['id'],
                        username=user_data['username'],
                        email=user_data['email'],
                        role=user_data['role'],
                        created_at=user_data['created_at'],
                        last_login=user_data['last_login']
                    )
            return None
        except Exception as e:
            logger.error(f"Session validation failed: {e}")
            return None
    
    def delete_session(self, session_id: str):
        """Delete user session"""
        with self._get_db_connection() as conn:
            conn.execute(
                "DELETE FROM user_sessions WHERE session_id = ?",
                (session_id,)
            )

class JWTManager:
    """JWT token management for stateless authentication"""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
    
    def create_token(self, user: User, expires_hours: int = 24) -> str:
        """Create JWT token"""
        payload = {
            'user_id': user.id,
            'username': user.username,
            'role': user.role,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=expires_hours),
            'iat': datetime.datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def validate_token(self, token: str) -> Optional[Dict]:
        """Validate JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            logger.error("JWT token expired")
            return None
        except jwt.InvalidTokenError:
            logger.error("Invalid JWT token")
            return None