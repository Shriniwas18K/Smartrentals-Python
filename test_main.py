import pytest
import json
import os
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, ANY
from cryptography.fernet import Fernet
from uuid import uuid4

# Import the FastAPI app from main.py
from main import app, get_embeddings, validate_token, cipher, generate_token, get_recommendations, require_auth

# Create a test client
client = TestClient(app)

# Test data
TEST_PHONE = "9999999999"
TEST_USERNAME = "testuser"
TEST_PASSWORD = "password123"
TEST_EMAIL = "test@example.com"
TEST_PROPERTY_ID = str(uuid4())

@pytest.fixture
def mock_db_connection():
    """Mock database connection and cursor"""
    with patch('main.connection') as mock_conn, patch('main.cur') as mock_cur:
        # Configure the mock cursor fetchone/fetchall methods
        mock_cur.fetchone.return_value = None
        mock_cur.fetchall.return_value = []
        
        yield mock_conn, mock_cur

@pytest.fixture
def mock_pinecone():
    """Mock Pinecone vector database"""
    with patch('main.index') as mock_index:
        # Configure mock methods
        mock_index.upsert.return_value = None
        mock_index.delete.return_value = None
        
        # Mock query response
        mock_match = MagicMock()
        mock_match.score = 0.95
        mock_match.metadata = {"property": json.dumps({
            "property_name": "Test Property",
            "property_types": "2 Bedroom",
            "rent_per_person": 1000,
            "address": "123 Test St"
        })}
        
        mock_response = MagicMock()
        mock_response.matches = [mock_match]
        mock_index.query.return_value = mock_response
        
        yield mock_index

@pytest.fixture
def mock_model():
    """Mock sentence transformer model"""
    with patch('main.model') as mock_model:
        mock_model.encode.return_value = [0.1, 0.2, 0.3]
        yield mock_model

@pytest.fixture
def valid_auth_token():
    """Generate a valid authentication token for testing"""
    token_data = {
        "phone": TEST_PHONE,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    token_json = json.dumps(token_data).encode()
    return cipher.encrypt(token_json).decode('utf-8')

@pytest.fixture
def expired_auth_token():
    """Generate an expired authentication token for testing"""
    expired_time = datetime.now() - timedelta(minutes=20)  # 20 minutes ago
    token_data = {
        "phone": TEST_PHONE,
        "timestamp": expired_time.strftime("%Y-%m-%d %H:%M:%S")
    }
    token_json = json.dumps(token_data).encode()
    return cipher.encrypt(token_json).decode('utf-8')

@pytest.fixture
def mock_require_auth():
    """Mock the require_auth dependency"""
    with patch('main.require_auth', return_value={"phone": TEST_PHONE}):
        yield

"""
Unit Tests
"""

def test_generate_token():
    """Test token generation"""
    token = generate_token(TEST_PHONE)
    assert token is not None
    assert isinstance(token, str)
    
    # Verify token can be decrypted
    decrypted = cipher.decrypt(token.encode())
    token_data = json.loads(decrypted.decode())
    
    assert "phone" in token_data
    assert token_data["phone"] == TEST_PHONE
    assert "timestamp" in token_data

def test_validate_token_valid(valid_auth_token):
    """Test token validation with valid token"""
    user_data = validate_token(valid_auth_token)
    assert user_data["phone"] == TEST_PHONE

def test_validate_token_expired(expired_auth_token):
    """Test token validation with expired token"""
    with pytest.raises(Exception):
        validate_token(expired_auth_token)

def test_get_embeddings(mock_model):
    """Test embedding generation"""
    result = get_embeddings("test query")
    assert result == [0.1, 0.2, 0.3]
    mock_model.encode.assert_called_once()

"""
API Tests
"""

def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "version" in data["data"]

def test_health_check(mock_db_connection, mock_pinecone):
    """Test health check endpoint"""
    mock_conn, mock_cur = mock_db_connection
    mock_cur.fetchone.return_value = [1]  # DB returns 1 for SELECT 1
    
    response = client.get("/health/")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["database"] == "connected"
    assert data["data"]["vector_database"] == "connected"

def test_signup_new_user(mock_db_connection):
    """Test user signup with new user"""
    mock_conn, mock_cur = mock_db_connection
    mock_cur.fetchone.return_value = None  # No existing user
    
    user_data = {
        "phone": TEST_PHONE,
        "username": TEST_USERNAME,
        "password": TEST_PASSWORD,
        "email": TEST_EMAIL
    }
    
    response = client.post("/signup/", json=user_data)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "User created successfully" in data["message"]
    
    # Verify DB operations
    mock_cur.execute.assert_any_call("SELECT * FROM credentials WHERE phone = %s", (TEST_PHONE,))
    mock_cur.execute.assert_any_call(
        "INSERT INTO credentials (phone, username, password, email, created_on) VALUES (%s, %s, %s, %s, %s)",
        (TEST_PHONE, TEST_USERNAME, TEST_PASSWORD, TEST_EMAIL, ANY)
    )
    mock_conn.commit.assert_called()

def test_signup_existing_user(mock_db_connection):
    """Test user signup with existing user"""
    mock_conn, mock_cur = mock_db_connection
    mock_cur.fetchone.return_value = ["existing_data"]  # Existing user
    
    user_data = {
        "phone": TEST_PHONE,
        "username": TEST_USERNAME,
        "password": TEST_PASSWORD
    }
    
    response = client.post("/signup/", json=user_data)
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert "User already exists" in data["message"]

def test_signin_valid_credentials(mock_db_connection):
    """Test user signin with valid credentials"""
    mock_conn, mock_cur = mock_db_connection
    mock_cur.fetchone.return_value = [TEST_USERNAME]  # User exists with username
    
    login_data = {
        "phone": TEST_PHONE,
        "password": TEST_PASSWORD
    }
    
    response = client.post("/signin/", json=login_data)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "Login successful" in data["message"]
    assert "token" in data["data"]
    assert data["data"]["username"] == TEST_USERNAME

def test_signin_invalid_credentials(mock_db_connection):
    """Test user signin with invalid credentials"""
    mock_conn, mock_cur = mock_db_connection
    mock_cur.fetchone.return_value = None  # No user with credentials
    
    login_data = {
        "phone": TEST_PHONE,
        "password": "wrong_password"
    }
    
    response = client.post("/signin/", json=login_data)
    assert response.status_code == 401
    data = response.json()
    assert data["success"] is False
    assert "Invalid" in data["message"]

def test_verify_token_valid(valid_auth_token):
    """Test token verification with valid token"""
    response = client.get(f"/verify-token/?token={valid_auth_token}")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "Token is valid" in data["message"]
    assert data["data"]["phone"] == TEST_PHONE

def test_verify_token_invalid(expired_auth_token):
    """Test token verification with invalid token"""
    response = client.get(f"/verify-token/?token={expired_auth_token}")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert "expired" in data["message"].lower() or "invalid" in data["message"].lower()

def test_create_property(mock_db_connection, mock_pinecone, valid_auth_token):
    """Test property creation"""
    with patch('main.require_auth', return_value={"phone": TEST_PHONE}):
        property_data = {
            "property_name": "Test Apartment",
            "property_types": "2 Bedroom",
            "security": "Gated Community",
            "parking_type": "Free Dedicated Parking",
            "lease_term": "12 Months with extension",
            "background": "Standard",
            "furnish_type": "Fully Furnished",
            "rent_per_person": 1000,
            "wifi_facility": "Available",
            "address": "123 Test Street",
            "city": "Test City",
            "state": "Test State",
            "zip_code": "12345"
        }
        
        response = client.post(
            "/properties/",
            json=property_data,
            headers={"Authorization": f"Bearer {valid_auth_token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "Property posted successfully" in data["message"]
        assert "property_id" in data["data"]

def test_get_user_properties(mock_db_connection, valid_auth_token):
    """Test getting properties for a user"""
    mock_conn, mock_cur = mock_db_connection
    
    # Mock property data
    property_data = {
        "property_name": "Test Apartment",
        "property_types": "2 Bedroom",
        "rent_per_person": 1000
    }
    
    mock_cur.fetchall.return_value = [
        (
            TEST_PROPERTY_ID, 
            json.dumps(property_data), 
            datetime.now(), 
            datetime.now()
        )
    ]
    
    response = client.get(f"/properties/?token={valid_auth_token}")
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "Retrieved" in data["message"]
    assert len(data["data"]["properties"]) == 1
    assert data["data"]["properties"][0]["property_name"] == "Test Apartment"

def test_search_properties(mock_pinecone):
    """Test property search"""
    search_query = {
        "location": "123 Main St",
        "property_type": "2 Bedroom",
        "min_rent": 800,
        "max_rent": 1500,
        "top_k": 5
    }
    
    response = client.post("/search/", json=search_query)
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "Found" in data["message"]
    assert "properties" in data["data"]

def test_get_property_details(mock_db_connection):
    """Test getting details for a specific property"""
    mock_conn, mock_cur = mock_db_connection
    
    # Mock property data
    property_data = {
        "property_name": "Test Apartment",
        "property_types": "2 Bedroom",
        "rent_per_person": 1000
    }
    
    # First query returns property data
    mock_cur.fetchone.side_effect = [
        (json.dumps(property_data), TEST_PHONE, datetime.now(), datetime.now()),
        (TEST_USERNAME,)  # Second fetchone for owner username
    ]
    
    response = client.get(f"/property/{TEST_PROPERTY_ID}/")
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["property_name"] == "Test Apartment"
    assert data["data"]["owner_username"] == TEST_USERNAME

def test_update_property(mock_db_connection, mock_pinecone):
    """Test property update"""
    mock_conn, mock_cur = mock_db_connection
    
    with patch('main.require_auth', return_value={"phone": TEST_PHONE}):
        # Mock current property data
        current_data = {
            "property_name": "Old Name",
            "property_types": "1 Bedroom",
            "security": "Not Applicable",
            "parking_type": "No Parking",
            "lease_term": "Month to Month",
            "background": "Standard",
            "furnish_type": "Unfurnished",
            "rent_per_person": 800,
            "wifi_facility": "Not Available",
            "address": "Old Address",
            "city": "Old City",
            "state": "Old State",
            "zip_code": "99999"
        }
        
        mock_cur.fetchone.return_value = (json.dumps(current_data),)
        
        # Update request
        update_data = {
            "property_name": "New Name",
            "rent_per_person": 900,
            "wifi_facility": "Available"
        }
        
        response = client.put(
            f"/properties/{TEST_PROPERTY_ID}/",
            json=update_data,
            headers={"Authorization": f"Bearer dummy-token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "Property updated successfully" in data["message"]
        
        # Check that the DB was updated
        mock_cur.execute.assert_any_call(
            "UPDATE properties SET property_data = %s, updated_on = %s WHERE property_id = %s",
            (ANY, ANY, TEST_PROPERTY_ID)
        )
        
        # Check vector DB was updated
        mock_pinecone.upsert.assert_called_once()

def test_delete_property(mock_db_connection, mock_pinecone):
    """Test property deletion"""
    mock_conn, mock_cur = mock_db_connection
    
    with patch('main.require_auth', return_value={"phone": TEST_PHONE}):
        # Mock property exists
        mock_cur.fetchone.return_value = (1,)
        
        response = client.delete(
            f"/properties/{TEST_PROPERTY_ID}/",
            headers={"Authorization": f"Bearer dummy-token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "Property deleted successfully" in data["message"]
        
        # Check that the DB deletion was called
        mock_cur.execute.assert_any_call(
            "DELETE FROM properties WHERE property_id = %s",
            (TEST_PROPERTY_ID,)
        )
        
        # Check vector DB deletion was called
        mock_pinecone.delete.assert_called_once_with(ids=[TEST_PROPERTY_ID])

if __name__ == "__main__":
    pytest.main()
