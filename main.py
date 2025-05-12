from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from pydantic import BaseModel, Field, EmailStr, validator
from typing import Literal, List, Optional, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4
import random
import psycopg2
import json
import os
import logging
from sentence_transformers import SentenceTransformer
from functools import lru_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("smartrentals")

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Smartrentals API",
    description="Property recommendation system with semantic search capabilities",
    version="1.0.0",
    contact={
        "name": "Shriniwas Kulkarni",
        "email": "kshriniwas180205@gmail.com",
        "url": "https://github.com/Shriniwas18K",
    }
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Generate a key for token encryption
key = Fernet.generate_key()
cipher = Fernet(key)

# Environment variables
DATABASE_URL = os.getenv('DATABASE_URL')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
TOKEN_EXPIRY_MINUTES = int(os.getenv('TOKEN_EXPIRY_MINUTES', '10'))

'''********************************************************************
                        Database connections
********************************************************************'''

try:
    connection = psycopg2.connect(
        DATABASE_URL
    )
    cur = connection.cursor()
    logger.info("Successfully connected to PostgreSQL database")

except (Exception, psycopg2.Error) as error:
    logger.error(f"Error while connecting to PostgreSQL: {error}")
    raise

# Create necessary database tables if they don't exist
cur.execute(
    '''
    CREATE TABLE IF NOT EXISTS credentials(
        phone VARCHAR(10) PRIMARY KEY,
        username VARCHAR(50) NOT NULL,
        password VARCHAR(100) NOT NULL,
        email VARCHAR(100),
        created_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    '''
)

cur.execute(
    '''
    CREATE TABLE IF NOT EXISTS transactions (
        transaction_id SERIAL PRIMARY KEY,
        transaction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        phone VARCHAR(10) REFERENCES credentials(phone),
        description VARCHAR(100),
        ip_address VARCHAR(45)
    )
    '''
)

cur.execute(
    '''
    CREATE TABLE IF NOT EXISTS properties (
        property_id UUID PRIMARY KEY,
        owner_phone VARCHAR(10) REFERENCES credentials(phone),
        property_data JSONB NOT NULL,
        created_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    '''
)

connection.commit()
logger.info("Database tables created or verified")

# Initialize Pinecone vector database
INDEX_NAME = 'pgrecommendervectordatabaseindex'
INDEX_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2 model

try:
    pinecone = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists, create if it doesn't
    existing_indexes = [index.name for index in pinecone.list_indexes()]
    
    if INDEX_NAME not in existing_indexes:
        logger.info(f"Creating new Pinecone index: {INDEX_NAME}")
        pinecone.create_index(
            name=INDEX_NAME, 
            dimension=INDEX_DIMENSION,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    
    index = pinecone.Index(INDEX_NAME)
    logger.info(f"Successfully connected to Pinecone index: {INDEX_NAME}")
    
except Exception as e:
    logger.error(f"Error initializing Pinecone: {e}")
    raise

# Load the sentence transformer model
@lru_cache(maxsize=1)
def get_model():
    """Load and cache the sentence transformer model"""
    logger.info("Loading sentence transformer model")
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model = get_model()

'''***********************************************************
                     Validation models
***********************************************************'''

class Auth(BaseModel):
    """User authentication model"""
    phone: str
    username: str
    password: str
    email: Optional[EmailStr] = None
    
    @validator('phone')
    def validate_phone(cls, v):
        if not v.isdigit() or len(v) != 10:
            raise ValueError('Phone number must be 10 digits')
        return v
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 6:
            raise ValueError('Password must be at least 6 characters')
        return v

class LoginRequest(BaseModel):
    """Login request model"""
    phone: str
    password: str
    
    @validator('phone')
    def validate_phone(cls, v):
        if not v.isdigit() or len(v) != 10:
            raise ValueError('Phone number must be 10 digits')
        return v

class Property(BaseModel):
    """Property details model with strict validation"""
    property_name: str = Field(..., min_length=3, max_length=100)
    property_types: Literal['1 Bedroom', '2 Bedroom', '3 Bedroom', '4 Bedroom', 'Studio']
    security: Literal['Not Applicable', 'Gated Community', 'Security Guard']
    parking_type: Literal['No Parking', 'Nearby Paid Parking', 'On-Street Parking', 
                       'Paid Dedicated Parking', 'Free Dedicated Parking']
    lease_term: Literal['Month to Month', '6 Months', '12 Months no extension', 
                     '12 Months with extension', 'Multi-Year']
    background: Literal['Negative', 'Further Review Required', 'Neutral or Mixed', 
                     'Standard', 'Positive']
    furnish_type: Literal['Fully Furnished', 'Partially Furnished', 'Unfurnished']
    rent_per_person: int = Field(..., gt=0)
    wifi_facility: Literal['Available', 'Not Available']
    address: str = Field(..., min_length=10)
    city: str
    state: str
    zip_code: str
    description: Optional[str] = None
    amenities: Optional[List[str]] = None

class PropertyUpdateRequest(BaseModel):
    """Property update model with optional fields"""
    property_name: Optional[str] = None
    property_types: Optional[Literal['1 Bedroom', '2 Bedroom', '3 Bedroom', '4 Bedroom', 'Studio']] = None
    security: Optional[Literal['Not Applicable', 'Gated Community', 'Security Guard']] = None
    parking_type: Optional[Literal['No Parking', 'Nearby Paid Parking', 'On-Street Parking', 
                             'Paid Dedicated Parking', 'Free Dedicated Parking']] = None
    lease_term: Optional[Literal['Month to Month', '6 Months', '12 Months no extension', 
                           '12 Months with extension', 'Multi-Year']] = None
    background: Optional[Literal['Negative', 'Further Review Required', 'Neutral or Mixed', 
                           'Standard', 'Positive']] = None
    furnish_type: Optional[Literal['Fully Furnished', 'Partially Furnished', 'Unfurnished']] = None
    rent_per_person: Optional[int] = None
    wifi_facility: Optional[Literal['Available', 'Not Available']] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    description: Optional[str] = None
    amenities: Optional[List[str]] = None

class SearchQuery(BaseModel):
    """Search query model"""
    location: str
    property_type: Optional[str] = None
    min_rent: Optional[int] = None
    max_rent: Optional[int] = None
    furnish_type: Optional[str] = None
    top_k: int = 10

class ApiResponse(BaseModel):
    """Standard API response model"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

'''*********************************************************
                        Utilities  	
*********************************************************'''

def generate_token(phone: str) -> str:
    """
    Generate a secure token for user authentication.
    
    Args:
        phone: User's phone number for identification
        
    Returns:
        Encrypted token string containing user phone and timestamp
    """
    # Include both timestamp and phone number in the token
    token_data = {
        "phone": phone,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Convert to JSON and encrypt
    token_json = json.dumps(token_data).encode()
    return cipher.encrypt(token_json).decode('utf-8')

def validate_token(token_value: str) -> dict:
    """
    Validate the authentication token and extract user information.
    
    Args:
        token_value: The encrypted token string
        
    Returns:
        Dictionary containing user phone if valid
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        # Decrypt the token
        decrypted_token = cipher.decrypt(token_value.encode())
        token_data = json.loads(decrypted_token.decode())
        
        # Extract information
        phone = token_data.get("phone")
        timestamp_str = token_data.get("timestamp")
        
        if not phone or not timestamp_str:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token format"
            )
        
        # Check token expiration
        token_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        current_time = datetime.now()
        
        if (current_time - token_time).seconds > (TOKEN_EXPIRY_MINUTES * 60):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired, please login again"
            )
        
        return {"phone": phone}
        
    except Exception as e:
        logger.error(f"Token validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )

def require_auth(token: str):
    """
    Dependency for routes that require authentication.
    
    Args:
        token: The authentication token
        
    Returns:
        Dictionary with user information if authenticated
    """
    return validate_token(token)

def get_embeddings(query: str or List[str]):
    """
    Convert query text to vector embeddings.
    
    Args:
        query: Text string or list of strings to convert
        
    Returns:
        Vector embeddings as numpy array
    """
    try:
        return model.encode(query).tolist()
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise

def upsert_to_vectordb(property_id: str, property_data: Property):
    """
    Store property in vector database with embeddings.
    
    Args:
        property_id: Unique ID for the property
        property_data: Property model data
    """
    try:
        # Create a searchable address string
        search_text = f"{property_data.address}, {property_data.city}, {property_data.state} {property_data.zip_code}"
        
        # Generate embeddings for the property
        embeddings = get_embeddings(search_text)
        
        # Prepare metadata for the vector
        prepped = [{
            'id': property_id,
            'values': embeddings,
            'metadata': {
                'property': json.dumps(property_data.dict())
            }
        }]
        
        # Upsert to Pinecone
        index.upsert(prepped)
        logger.info(f"Property {property_id} upserted to vector database")
        
    except Exception as e:
        logger.error(f"Error upserting to vector database: {e}")
        raise

def get_recommendations(search_query: SearchQuery) -> List[dict]:
    """
    Retrieve semantically similar properties based on search criteria.
    
    Args:
        search_query: Search parameters
        
    Returns:
        List of recommended properties
    """
    try:
        # Generate embeddings for the search location
        embed = get_embeddings(search_query.location)
        
        # Query the vector database
        res = index.query(
            vector=embed, 
            top_k=search_query.top_k, 
            include_metadata=True
        )
        
        # Process and filter results
        results = []
        for match in res.matches:
            property_data = json.loads(match.metadata["property"])
            
            # Apply filters if provided
            if (search_query.property_type and 
                property_data["property_types"] != search_query.property_type):
                continue
                
            if (search_query.min_rent is not None and 
                property_data["rent_per_person"] < search_query.min_rent):
                continue
                
            if (search_query.max_rent is not None and 
                property_data["rent_per_person"] > search_query.max_rent):
                continue
                
            if (search_query.furnish_type and 
                property_data["furnish_type"] != search_query.furnish_type):
                continue
            
            # Add score to property data
            property_data["similarity_score"] = match.score
            results.append(property_data)
        
        return results
        
    except Exception as e:
        logger.error(f"Error retrieving recommendations: {e}")
        raise

def log_transaction(phone: str, description: str, ip_address: str = "0.0.0.0"):
    """
    Log user transaction to database.
    
    Args:
        phone: User's phone number
        description: Transaction description
        ip_address: User's IP address
    """
    try:
        cur.execute(
            "INSERT INTO transactions (transaction_time, phone, description, ip_address) VALUES (%s, %s, %s, %s)",
            (datetime.now(), phone, description, ip_address)
        )
        connection.commit()
    except Exception as e:
        logger.error(f"Error logging transaction: {e}")
        connection.rollback()

'''***********************************************************
	          Authentication routes
***********************************************************'''

@app.post("/signup/", response_model=ApiResponse)
async def sign_up(request: Auth):
    """
    Register a new user account.
    
    Args:
        request: User registration details
        
    Returns:
        Success message or error details
    """
    try:
        # Check if user already exists
        cur.execute("SELECT * FROM credentials WHERE phone = %s", (request.phone,))
        if cur.fetchone():
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"success": False, "message": "User already exists, please sign in"}
            )
        
        # Insert new user
        cur.execute(
            "INSERT INTO credentials (phone, username, password, email, created_on) VALUES (%s, %s, %s, %s, %s)",
            (request.phone, request.username, request.password, request.email, datetime.now())
        )
        
        # Log transaction
        log_transaction(request.phone, "account_creation")
        
        connection.commit()
        logger.info(f"New user created: {request.phone}")
        
        return {
            "success": True,
            "message": "User created successfully",
            "data": {"username": request.username}
        }
        
    except Exception as e:
        connection.rollback()
        logger.error(f"Error in signup: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating user: {str(e)}"
        )
    
@app.post("/signin/", response_model=ApiResponse)
async def sign_in(request: LoginRequest):
    """
    Authenticate user and generate access token.
    
    Args:
        request: Login credentials
        
    Returns:
        Authentication token if successful
    """
    try:
        # Check if user exists with correct password
        cur.execute(
            "SELECT username FROM credentials WHERE phone = %s AND password = %s",
            (request.phone, request.password)
        )
        
        user = cur.fetchone()
        if not user:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "success": False, 
                    "message": "Invalid phone number or password"
                }
            )
        
        # Generate token
        token = generate_token(request.phone)
        
        # Log login transaction
        log_transaction(request.phone, "user_login")
        
        connection.commit()
        logger.info(f"User logged in: {request.phone}")
        
        return {
            "success": True,
            "message": "Login successful",
            "data": {
                "token": token,
                "username": user[0],
                "expires_in": f"{TOKEN_EXPIRY_MINUTES} minutes"
            }
        }
        
    except Exception as e:
        connection.rollback()
        logger.error(f"Error in signin: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during login: {str(e)}"
        )

@app.get("/verify-token/", response_model=ApiResponse)
async def verify_token(token: str):
    """
    Verify if a token is valid and not expired.
    
    Args:
        token: Authentication token
        
    Returns:
        Token validity status
    """
    try:
        user_data = validate_token(token)
        return {
            "success": True,
            "message": "Token is valid",
            "data": {"phone": user_data["phone"]}
        }
    except HTTPException as he:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "success": False,
                "message": he.detail,
                "data": None
            }
        )

'''*************************************************************************
	          Property posting and retrieval routes
************************************************************************'''

@app.post("/properties/", response_model=ApiResponse)
async def create_property(property_data: Property, auth: dict = Depends(require_auth)):
    """
    Create a new property listing.
    
    Args:
        property_data: Property details
        auth: Authenticated user information
        
    Returns:
        Success message with property ID
    """
    try:
        # Generate unique ID for property
        property_id = str(uuid4())
        
        # Store property in PostgreSQL
        cur.execute(
            "INSERT INTO properties (property_id, owner_phone, property_data, created_on, updated_on) VALUES (%s, %s, %s, %s, %s)",
            (property_id, auth["phone"], json.dumps(property_data.dict()), datetime.now(), datetime.now())
        )
        
        # Store property in vector database for recommendations
        upsert_to_vectordb(property_id, property_data)
        
        # Log transaction
        log_transaction(auth["phone"], f"property_creation:{property_id}")
        
        connection.commit()
        logger.info(f"Property created: {property_id} by user {auth['phone']}")
        
        return {
            "success": True,
            "message": "Property posted successfully",
            "data": {"property_id": property_id}
        }
        
    except Exception as e:
        connection.rollback()
        logger.error(f"Error creating property: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating property: {str(e)}"
        )

@app.get("/properties/", response_model=ApiResponse)
async def get_user_properties(token: str):
    """
    Get all properties owned by the authenticated user.
    
    Args:
        token: Authentication token
        
    Returns:
        List of user's properties
    """
    try:
        # Validate token
        user_data = validate_token(token)
        
        # Get properties for this user
        cur.execute(
            "SELECT property_id, property_data, created_on, updated_on FROM properties WHERE owner_phone = %s",
            (user_data["phone"],)
        )
        
        properties = []
        for row in cur.fetchall():
            property_data = json.loads(row[1])
            properties.append({
                "property_id": row[0],
                "created_on": row[2].isoformat(),
                "updated_on": row[3].isoformat(),
                **property_data
            })
        
        # Log transaction
        log_transaction(user_data["phone"], "property_listing_retrieval")
        
        return {
            "success": True,
            "message": f"Retrieved {len(properties)} properties",
            "data": {"properties": properties}
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving properties: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving properties: {str(e)}"
        )

@app.put("/properties/{property_id}/", response_model=ApiResponse)
async def update_property(
    property_id: str, 
    update_data: PropertyUpdateRequest, 
    auth: dict = Depends(require_auth)
):
    """
    Update an existing property listing.
    
    Args:
        property_id: ID of property to update
        update_data: Property data to update
        auth: Authenticated user information
        
    Returns:
        Success message
    """
    try:
        # Verify property exists and belongs to user
        cur.execute(
            "SELECT property_data FROM properties WHERE property_id = %s AND owner_phone = %s",
            (property_id, auth["phone"])
        )
        
        result = cur.fetchone()
        if not result:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "success": False,
                    "message": "Property not found or you don't have permission to update it"
                }
            )
        
        # Get current property data and update with new values
        current_data = json.loads(result[0])
        
        # Update only provided fields
        for field, value in update_data.dict(exclude_unset=True).items():
            if value is not None:
                current_data[field] = value
        
        # Update in database
        cur.execute(
            "UPDATE properties SET property_data = %s, updated_on = %s WHERE property_id = %s",
            (json.dumps(current_data), datetime.now(), property_id)
        )
        
        # Update in vector database
        property_model = Property(**current_data)
        upsert_to_vectordb(property_id, property_model)
        
        # Log transaction
        log_transaction(auth["phone"], f"property_update:{property_id}")
        
        connection.commit()
        logger.info(f"Property updated: {property_id} by user {auth['phone']}")
        
        return {
            "success": True,
            "message": "Property updated successfully",
            "data": {"property_id": property_id}
        }
        
    except Exception as e:
        connection.rollback()
        logger.error(f"Error updating property: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating property: {str(e)}"
        )

@app.delete("/properties/{property_id}/", response_model=ApiResponse)
async def delete_property(property_id: str, auth: dict = Depends(require_auth)):
    """
    Delete a property listing.
    
    Args:
        property_id: ID of property to delete
        auth: Authenticated user information
        
    Returns:
        Success message
    """
    try:
        # Verify property exists and belongs to user
        cur.execute(
            "SELECT 1 FROM properties WHERE property_id = %s AND owner_phone = %s",
            (property_id, auth["phone"])
        )
        
        if not cur.fetchone():
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "success": False,
                    "message": "Property not found or you don't have permission to delete it"
                }
            )
        
        # Delete from database
        cur.execute(
            "DELETE FROM properties WHERE property_id = %s",
            (property_id,)
        )
        
        # Delete from vector database
        try:
            index.delete(ids=[property_id])
        except Exception as e:
            logger.warning(f"Error deleting from vector DB (continuing anyway): {e}")
        
        # Log transaction
        log_transaction(auth["phone"], f"property_deletion:{property_id}")
        
        connection.commit()
        logger.info(f"Property deleted: {property_id} by user {auth['phone']}")
        
        return {
            "success": True,
            "message": "Property deleted successfully"
        }
        
    except Exception as e:
        connection.rollback()
        logger.error(f"Error deleting property: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting property: {str(e)}"
        )

@app.post("/search/", response_model=ApiResponse)
async def search_properties(search_query: SearchQuery):
    """
    Search properties using semantic search and filters.
    
    Args:
        search_query: Search and filter criteria
        
    Returns:
        List of matching properties
    """
    try:
        # Get recommendations based on search criteria
        recommendations = get_recommendations(search_query)
        
        logger.info(f"Search performed for '{search_query.location}', found {len(recommendations)} results")
        
        return {
            "success": True,
            "message": f"Found {len(recommendations)} matching properties",
            "data": {"properties": recommendations}
        }
        
    except Exception as e:
        logger.error(f"Error searching properties: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching properties: {str(e)}"
        )

@app.get("/property/{property_id}/", response_model=ApiResponse)
async def get_property_details(property_id: str):
    """
    Get detailed information about a specific property.
    
    Args:
        property_id: ID of property to retrieve
        
    Returns:
        Property details
    """
    try:
        # Get property from database
        cur.execute(
            "SELECT property_data, owner_phone, created_on, updated_on FROM properties WHERE property_id = %s",
            (property_id,)
        )
        
        result = cur.fetchone()
        if not result:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "success": False,
                    "message": "Property not found"
                }
            )
        
        property_data = json.loads(result[0])
        
        # Get owner username
        cur.execute(
            "SELECT username FROM credentials WHERE phone = %s",
            (result[1],)
        )
        
        owner = cur.fetchone()
        
        return {
            "success": True,
            "message": "Property details retrieved",
            "data": {
                "property_id": property_id,
                "owner_username": owner[0] if owner else "Unknown",
                "created_on": result[2].isoformat(),
                "updated_on": result[3].isoformat(),
                **property_data
            }
        }
        
    except Exception as e:
        logger.error(f"Error retrieving property details: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving property details: {str(e)}"
        )

'''*************************************************************************
                           Health and Status Routes
************************************************************************'''

@app.get("/", response_model=ApiResponse)
async def root():
    """API root endpoint returning service status"""
    return {
        "success": True,
        "message": "Smartrentals API is running",
        "data": {
            "version": "1.0.0",
            "status": "healthy"
        }
    }

@app.get("/health/", response_model=ApiResponse)
async def health_check():
    """
    System health check endpoint.
    
    Returns:
        Service status information
    """
    try:
        # Check database connection
        cur.execute("SELECT 1")
        db_status = "connected" if cur.fetchone() else "error"
        
        # Check vector database
        try:
            index.describe_index_stats()
            vector_db_status = "connected"
        except:
            vector_db_status = "error"
        
        return {
            "success": True,
            "message": "System health check completed",
            "data": {
                "database": db_status,
                "vector_database": vector_db_status,
                "api": "healthy",
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "message": "System health check failed",
                "data": {"error": str(e)}
            }
        )

# Application shutdown handler
@app.on_event("shutdown")
def shutdown_event():
    """Clean up resources when application shuts down"""
    logger.info("Application shutting down, cleaning up resources")
    if connection:
        cur.close()
        connection.close()
        logger.info("Database connection closed")

# Run the application with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)