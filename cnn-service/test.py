# test the ../predict.py file
from fastapi.testclient import TestClient
from predict import app
import io
from PIL import Image
client = TestClient(app)


def test_supported_classes():
    response = client.get("/supported_classes")
    assert response.status_code == 200
    data = response.json()
    assert "supported_classes" in data
    assert "warning" in data
    assert len(data["supported_classes"]) == 5

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_classify_image():
    # Create a dummy image for testing
    image = Image.new('L', (48, 48), color=128)  # Grayscale image
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    files = {'file': ('test/test.png', img_byte_arr, 'image/png')}
    response = client.post("/classify", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert "supported_classes" in data
    assert "warning" in data


if __name__ == "__main__":
    print('Running tests...')
    test_supported_classes()
    test_health()
    test_classify_image()
    print("All tests passed.")