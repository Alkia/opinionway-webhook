import requests
import json

# Define the URL of your Flask app
url = "http://localhost:8888/opinionway"

# Define test cases
test_cases = [
    {
        "input": {
            "transcript": "This is a great product!",
            "uid": "12345"
        },
        "expected_message": "Theme matched"
    },
    {
        "input": {
            "transcript": "This is the worst experience ever.",
            "uid": "67890"
        },
        "expected_message": "Theme matched"
    },
    {
        "input": {
            "transcript": "This is neutral feedback.",
            "uid": "11223"
        },
        "expected_message": "No theme matched"
    }
]

# Run tests
for i, test_case in enumerate(test_cases):
    response = requests.post(url, json=test_case["input"])
    
    if response.status_code == 200:
        data = response.json()
        print(f"Test Case {i + 1}:")
        print(f"Input: {json.dumps(test_case['input'], indent=2)}")
        print(f"Expected Message: {test_case['expected_message']}")
        print(f"Response: {json.dumps(data, indent=2)}")
        print("Passed\n" if data["message"] == test_case["expected_message"] else "Failed\n")
    else:
        print(f"Test Case {i + 1} failed with status code {response.status_code}\n")
