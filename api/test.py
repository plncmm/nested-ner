import requests
url = 'http://localhost:5000/predict'
r = requests.post(url, json = {'text': 'hola'})
print(r.text)
r.text.strip()