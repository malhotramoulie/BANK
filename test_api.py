import requests

url = "http://127.0.0.1:5000"
data = {
    "model_type": "logistic_regression",
    "dataset": "df1"
}

try:
    response = requests.post(url, json=data)
    response.raise_for_status()  # Raise an error for bad status codes
    json_response = response.json()  # Try to parse JSON response
    print(json_response)
except requests.exceptions.HTTPError as http_err:
    print(f"HTTP error occurred: {http_err}")  # Print HTTP error
except requests.exceptions.RequestException as req_err:
    print(f"Request error occurred: {req_err}")  # Print request error
except ValueError as json_err:
    print("Response is not in JSON format")
    print("Raw response text:", response.text)  # Print the raw text of the response
