import requests

url = "http://127.0.0.1:8000/rerank"

data = {
    "query": "What are the treatment options for cancer?",
    "documents": [
        "The hospital cafeteria provides breakfast.",
        "Cancer treatment may include surgery, chemotherapy and radiation therapy.",
        "The hospital has parking facilities.",
        "Patients with cancer should be referred for appropriate treatment.",
        "The emergency department is open 24 hours."
    ],
    "top_k": 3
}

response = requests.post(
    url,
    json=data
)

print(response.status_code)
print(response.json())