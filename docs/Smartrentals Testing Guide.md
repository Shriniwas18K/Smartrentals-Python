# Smartrentals Testing Guide

This document provides instructions for setting up and running tests for the Smartrentals API.

## Test Environment Setup

1. Create a test database:
   ```bash
   createdb test
   ```

2. Create a test environment file:
   - Copy the provided `.env.test` file to your project root
   - Update the values with your local test environment configuration

3. Install test dependencies:
   ```bash
   pip install pytest pytest-cov requests
   ```

## Running Tests

### Unit Tests

Run unit tests with coverage reporting:

```bash
pytest test_main.py -v --cov=main
```

### End-to-End Tests

These tests require the API to be running:

1. Start the API in test mode:
   ```bash
   DATABASE_URL=postgresql://postgres:password@localhost:5432/smartrentals_test \
   PINECONE_API_KEY=your-pinecone-api-key \
   uvicorn main:app --reload
   ```

2. In a separate terminal, run the E2E tests:
   ```bash
   pytest e2e_test.py -v
   ```

## Test Structure

The test suite includes:

### Unit Tests (`test_main.py`)

- Focuses on testing individual components
- Uses mocking for external dependencies
- Tests API endpoints with a test client

Key components tested:
- Authentication (token generation, validation)
- Property APIs (create, read, update, delete)
- Search functionality
- Error handling

### End-to-End Tests (`e2e_test.py`)

- Tests complete user flows
- Makes real HTTP requests to a running API
- Verifies integration between components

Key flows tested:
- User registration and authentication
- Property lifecycle (create, read, update, delete)
- Search with various filters

## Continuous Integration

To run tests in a CI environment:

```bash
# Run all tests
pytest

# Run with coverage reporting
pytest --cov=main

# Generate HTML coverage report
pytest --cov=main --cov-report=html
```

## Mock Environment Variables

The tests expect the following environment variables to be set:

```
TEST_API_URL=http://localhost:8000
TEST_DATABASE_URL=postgresql://postgres:password@localhost:5432/smartrentals_test
TEST_PINECONE_API_KEY=your-pinecone-api-key
```

## Test Data

- Test users are created with the prefix `testuser`
- Test properties are created with the prefix `Test Apartment`
- E2E tests create a single user and property for testing

## Troubleshooting

If you encounter issues:

1. Check if the test database exists and is accessible
2. Verify that the API is running when executing E2E tests
3. Check if the Pinecone API key is valid
4. Run tests with increased verbosity: `pytest -vv`

For more detailed logging, modify the log level in the test files.
