import os

if os.getenv('ENVIRONMENT') == 'container':
    print("Running inside a Docker container")
else:
    print("Running on the local machine")


import requests

ride = {
    "PULocationID": 10,
    "DOLocationID": 50,
    "trip_distance": 40
}

url = 'http://127.0.0.1:9696/predict'
response = requests.post(url, json=ride)

# Print the raw response in case of an error
try:
    print(response.json())
except requests.exceptions.JSONDecodeError:
    print("Response content:", response.text)
