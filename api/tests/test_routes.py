from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_home_route():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "The server is up and running!"}


def test_health_route():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
