import pytest
import requests
import json
import time
import logging
from dotenv import load_dotenv
import os

# Load environment variables from .env.test file
load_dotenv(".env.test")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("e2e_tests")

# Test configuration
API_URL = os.getenv("TEST_API_URL", "http://localhost:8000")
TEST_USER = {
    "phone": "7777777777",
    "username": "e2etestuser",
    "password": "testpass123",
    "email": "e2etest@example.com"
}
TEST_PROPERTY = {
    "property_name": "E2E Test Apartment",
    "property_types": "2 Bedroom",
    "security": "Gated Community",
    "parking_type": "Free Dedicated Parking",
    "lease_term": "12 Months with extension",
    "background": "Standard",
    "furnish_type": "Fully Furnished",
    "rent_per_person": 1200,
    "wifi_facility": "Available",
    "address": "456 E2E Test Street",
    "city": "Test City",
    "state": "Test State",
    "zip_code": "54321",
    "description": "A beautiful apartment for end-to-end testing",
    "amenities": ["Gym", "Pool", "Pet Friendly"]
}

# Test state
test_state = {
    "auth_token": None,
    "property_id": None
}

def request_with_logging(method, endpoint, **kwargs):
    """Make HTTP request with logging"""
    url = f"{API_URL}{endpoint}"
    logger.info(f"Making {method} request to {url}")
    
    if "json" in kwargs:
        logger.info(f"Request body: {json.dumps(kwargs['json'], indent=2)}")
        
    if test_state["auth_token"] and "headers" not in kwargs:
        kwargs["headers"] = {"Authorization": f"Bearer {test_state['auth_token']}"}
    
    response = requests.request(method, url, **kwargs)
    
    try:
        response_data = response.json()
        logger.info(f"Response status: {response.status_code}")
        logger.info(f"Response body: {json.dumps(response_data, indent=2)}")
        return response
    except:
        logger.error(f"Failed to parse response as JSON: {response.text}")
        return response

class TestSmartrentalsE2E:
    """End-to-end tests for the Smartrentals API"""
    
    def setup_class(self):
        """Setup before all tests"""
        logger.info("Starting E2E test suite")
        # Health check to ensure API is running
        response = request_with_logging("GET", "/health/")
        assert response.status_code == 200, "API is not running"
    
    def teardown_class(self):
        """Cleanup after all tests"""
        logger.info("Completed E2E test suite")
    
    def test_01_signup_user(self):
        """Test user signup flow"""
        # First try to sign in (in case user exists from previous test run)
        response = request_with_logging("POST", "/signin/", json={
            "phone": TEST_USER["phone"],
            "password": TEST_USER["password"]
        })
        
        if response.status_code == 200 and response.json()["success"]:
            # User exists, capture token
            test_state["auth_token"] = response.json()["data"]["token"]
            logger.info("User already exists, signed in successfully")
            return
        
        # User doesn't exist, create new
        response = request_with_logging("POST", "/signup/", json=TEST_USER)
        assert response.status_code == 200
        assert response.json()["success"] is True
        
        # Now sign in
        response = request_with_logging("POST", "/signin/", json={
            "phone": TEST_USER["phone"],
            "password": TEST_USER["password"]
        })
        assert response.status_code == 200
        assert response.json()["success"] is True
        
        # Store token for subsequent tests
        test_state["auth_token"] = response.json()["data"]["token"]
    
    def test_02_verify_token(self):
        """Test token verification"""
        assert test_state["auth_token"], "No auth token available"
        
        response = request_with_logging(
            "GET", 
            f"/verify-token/?token={test_state['auth_token']}"
        )
        assert response.status_code == 200
        assert response.json()["success"] is True
    
    def test_03_create_property(self):
        """Test property creation"""
        assert test_state["auth_token"], "No auth token available"
        
        response = request_with_logging(
            "POST", 
            "/properties/", 
            json=TEST_PROPERTY
        )
        assert response.status_code == 200
        assert response.json()["success"] is True
        
        # Store property ID for subsequent tests
        test_state["property_id"] = response.json()["data"]["property_id"]
    
    def test_04_get_user_properties(self):
        """Test retrieving user properties"""
        assert test_state["auth_token"], "No auth token available"
        
        response = request_with_logging(
            "GET", 
            f"/properties/?token={test_state['auth_token']}"
        )
        assert response.status_code == 200
        assert response.json()["success"] is True
        
        # Verify our test property is in the list
        properties = response.json()["data"]["properties"]
        assert any(p["property_id"] == test_state["property_id"] for p in properties)
    
    def test_05_get_property_details(self):
        """Test getting specific property details"""
        assert test_state["property_id"], "No property ID available"
        
        response = request_with_logging(
            "GET", 
            f"/property/{test_state['property_id']}/"
        )
        assert response.status_code == 200
        assert response.json()["success"] is True
        
        # Verify property details
        property_data = response.json()["data"]
        assert property_data["property_name"] == TEST_PROPERTY["property_name"]
        assert property_data["rent_per_person"] == TEST_PROPERTY["rent_per_person"]
    
    def test_06_update_property(self):
        """Test updating property"""
        assert test_state["auth_token"], "No auth token available"
        assert test_state["property_id"], "No property ID available"
        
        update_data = {
            "property_name": "Updated E2E Test Apartment",
            "rent_per_person": 1300,
            "description": "Updated description from E2E test"
        }
        
        response = request_with_logging(
            "PUT", 
            f"/properties/{test_state['property_id']}/", 
            json=update_data
        )
        assert response.status_code == 200
        assert response.json()["success"] is True
        
        # Verify update was successful
        response = request_with_logging(
            "GET", 
            f"/property/{test_state['property_id']}/"
        )
        property_data = response.json()["data"]
        assert property_data["property_name"] == update_data["property_name"]
        assert property_data["rent_per_person"] == update_data["rent_per_person"]
    
    def test_07_search_properties(self):
        """Test property search"""
        # Add a short delay to ensure property is indexed
        time.sleep(1)
        
        search_query = {
            "location": "Test City",  # City of our test property
            "top_k": 10
        }
        
        response = request_with_logging(
            "POST", 
            "/search/", 
            json=search_query
        )
        assert response.status_code == 200
        assert response.json()["success"] is True
        
        # Our property should be in the results
        properties = response.json()["data"]["properties"]
        matching = [p for p in properties if "Updated E2E Test Apartment" in p.get("property_name", "")]
        assert matching, "Our test property was not found in search results"
    
    def test_08_filtered_search(self):
        """Test property search with filters"""
        search_query = {
            "location": "Test City",
            "property_type": "2 Bedroom",
            "min_rent": 1200,
            "max_rent": 1400,
            "furnish_type": "Fully Furnished",
            "top_k": 10
        }
        
        response = request_with_logging(
            "POST", 
            "/search/", 
            json=search_query
        )
        assert response.status_code == 200
        assert response.json()["success"] is True
    
    def test_09_delete_property(self):
        """Test property deletion"""
        assert test_state["auth_token"], "No auth token available"
        assert test_state["property_id"], "No property ID available"
        
        response = request_with_logging(
            "DELETE", 
            f"/properties/{test_state['property_id']}/"
        )
        assert response.status_code == 200
        assert response.json()["success"] is True
        
        # Verify property is gone
        response = request_with_logging(
            "GET", 
            f"/property/{test_state['property_id']}/"
        )
        assert response.status_code == 404 or not response.json()["success"]

if __name__ == "__main__":
    pytest.main(["-v"])
