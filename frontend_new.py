import requests

files = {"file": open("image.png", "rb")}
res = requests.post("https://<ganeshkende>.huggingface.co/spaces/<pomegranate-disease-detection-app-backend>/predict", files=files)
prediction = res.json()
