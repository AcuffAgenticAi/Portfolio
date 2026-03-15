import requests

url = "http://localhost:8000/detect"

payload = {
 "price_ratio": [1.0, 1.1, 5.0]
}

for i in range(1000):

    requests.post(url,json=payload)