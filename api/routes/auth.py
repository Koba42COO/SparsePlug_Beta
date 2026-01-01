"""Authentication Routes."""

from datetime import datetime, timedelta
from typing import Annotated, Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException, status, Header
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field, ConfigDict

from api.config import settings

router = APIRouter()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login", auto_error=False)


# ============== Schemas ==============

class UserCreate(BaseModel):
    """User registration schema."""
    email: EmailStr
    password: str = Field(..., min_length=8, description="Password (min 8 characters)")


class UserLogin(BaseModel):
    """User login schema."""
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """User response schema."""
    id: str
    email: str
    tier: str
    is_active: bool
    is_verified: bool
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class Token(BaseModel):
    """JWT Token response."""
    access_token: str
    token_type: str = "bearer"
    expires_at: datetime


class APIKeyResponse(BaseModel):
    """API Key response."""
    api_key: str
    name: str
    created_at: datetime
    permissions: list[str]


class APIKeyCreate(BaseModel):
    """API Key creation request."""
    name: str = Field(..., min_length=1, max_length=50)
    permissions: list[str] = Field(default=["read", "write", "optimize"])


# ============== Fake Database (Replace with real DB) ==============

# In-memory storage for demo (replace with SQLAlchemy models)
fake_users_db: dict = {}
fake_api_keys_db: dict = {}


class User:
    """User model (placeholder)."""
    def __init__(self, id: str, email: str, hashed_password: str, tier: str = "free"):
        self.id = id
        self.email = email
        self.hashed_password = hashed_password
        self.tier = tier
        self.is_active = True
        self.is_verified = False
        self.created_at = datetime.utcnow()
        self.api_keys: list[str] = []


# ============== Helper Functions ==============

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> tuple[str, datetime]:
    """Create a JWT access token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.jwt_access_token_expire_minutes)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    
    return encoded_jwt, expire


def generate_api_key() -> str:
    """Generate a new API key."""
    return f"ps_live_{uuid.uuid4().hex}"


def get_user_by_email(email: str) -> Optional[User]:
    """Get user by email."""
    return fake_users_db.get(email)


def get_user_by_id(user_id: str) -> Optional[User]:
    """Get user by ID."""
    for user in fake_users_db.values():
        if user.id == user_id:
            return user
    return None


async def get_current_user(
    token: Annotated[Optional[str], Depends(oauth2_scheme)] = None,
    x_api_key: Annotated[Optional[str], Header()] = None,
) -> User:
    """
    Get current authenticated user.
    
    Supports both JWT Bearer token and API key authentication.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # Try API key first
    if x_api_key:
        api_key_data = fake_api_keys_db.get(x_api_key)
        if api_key_data:
            user = get_user_by_id(api_key_data["user_id"])
            if user:
                return user
        raise credentials_exception
    
    # Try JWT token
    if token:
        try:
            payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
            user_id: str = payload.get("sub")
            if user_id is None:
                raise credentials_exception
        except JWTError:
            raise credentials_exception
        
        user = get_user_by_id(user_id)
        if user is None:
            raise credentials_exception
        
        return user
    
    raise credentials_exception


# Dependency for requiring auth
CurrentUser = Annotated[User, Depends(get_current_user)]


# ============== Endpoints ==============

@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate):
    """
    Register a new user.
    
    Creates a new user account and returns an access token.
    """
    # Check if user exists
    if get_user_by_email(user_data.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create user
    user_id = str(uuid.uuid4())
    hashed_password = get_password_hash(user_data.password)
    user = User(id=user_id, email=user_data.email, hashed_password=hashed_password)
    fake_users_db[user_data.email] = user
    
    # Create token
    access_token, expires_at = create_access_token(data={"sub": user_id})
    
    return Token(access_token=access_token, expires_at=expires_at)


@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Login and get access token.
    
    Use email as username.
    """
    user = get_user_by_email(form_data.username)
    
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token, expires_at = create_access_token(data={"sub": user.id})
    
    return Token(access_token=access_token, expires_at=expires_at)


@router.post("/refresh", response_model=Token)
async def refresh_token(current_user: CurrentUser):
    """Refresh the access token."""
    access_token, expires_at = create_access_token(data={"sub": current_user.id})
    return Token(access_token=access_token, expires_at=expires_at)


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: CurrentUser):
    """Get current user information."""
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        tier=current_user.tier,
        is_active=current_user.is_active,
        is_verified=current_user.is_verified,
        created_at=current_user.created_at,
    )


@router.get("/api-keys", response_model=list[APIKeyResponse])
async def list_api_keys(current_user: CurrentUser):
    """List all API keys for current user."""
    keys = []
    for key, data in fake_api_keys_db.items():
        if data["user_id"] == current_user.id:
            keys.append(APIKeyResponse(
                api_key=key[:12] + "..." + key[-4:],  # Show partial key
                name=data["name"],
                created_at=data["created_at"],
                permissions=data["permissions"],
            ))
    return keys


@router.post("/api-keys", response_model=APIKeyResponse)
async def create_api_key(key_data: APIKeyCreate, current_user: CurrentUser):
    """
    Create a new API key.
    
    **Important:** The full API key is only shown once. Save it securely!
    """
    # Check limit (5 keys per user)
    user_keys = [k for k, v in fake_api_keys_db.items() if v["user_id"] == current_user.id]
    if len(user_keys) >= 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum number of API keys (5) reached"
        )
    
    # Generate key
    api_key = generate_api_key()
    created_at = datetime.utcnow()
    
    # Store key data
    fake_api_keys_db[api_key] = {
        "user_id": current_user.id,
        "name": key_data.name,
        "permissions": key_data.permissions,
        "created_at": created_at,
    }
    
    # Return full key (only time it's shown!)
    return APIKeyResponse(
        api_key=api_key,  # Full key shown only on creation
        name=key_data.name,
        created_at=created_at,
        permissions=key_data.permissions,
    )


@router.delete("/api-keys/{key_prefix}")
async def revoke_api_key(key_prefix: str, current_user: CurrentUser):
    """Revoke an API key by its prefix."""
    # Find key by prefix
    for key in list(fake_api_keys_db.keys()):
        if key.startswith(key_prefix) and fake_api_keys_db[key]["user_id"] == current_user.id:
            del fake_api_keys_db[key]
            return {"message": "API key revoked"}
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="API key not found"
    )


@router.post("/api-key/regenerate", response_model=APIKeyResponse)
async def regenerate_api_key(current_user: CurrentUser):
    """
    Regenerate the primary API key.
    
    This invalidates the old key immediately.
    """
    # Remove existing keys
    keys_to_remove = [k for k, v in fake_api_keys_db.items() if v["user_id"] == current_user.id]
    for key in keys_to_remove:
        del fake_api_keys_db[key]
    
    # Generate new key
    api_key = generate_api_key()
    created_at = datetime.utcnow()
    
    fake_api_keys_db[api_key] = {
        "user_id": current_user.id,
        "name": "primary",
        "permissions": ["read", "write", "optimize"],
        "created_at": created_at,
    }
    
    return APIKeyResponse(
        api_key=api_key,
        name="primary",
        created_at=created_at,
        permissions=["read", "write", "optimize"],
    )
