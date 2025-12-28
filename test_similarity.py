import requests
import numpy as np

# Get embeddings for similar products
response = requests.post('http://localhost:8000/embed', json={
    'texts': [
        'HP 80A INK CARTRIDGE BLACK',
        'HP 36A INK CARTRIDGE BLACK',
        'Epson 664 Cyan Ink'
    ]
})

data = response.json()
embeddings = np.array(data['embeddings'])

# Calculate cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# HP 80A vs HP 36A (should be very similar - both HP black ink)
sim1 = cosine_similarity(embeddings[0], embeddings[1])
print(f"HP 80A vs HP 36A similarity: {sim1:.4f}")

# HP 80A vs Epson Cyan (should be less similar - different brand & color)
sim2 = cosine_similarity(embeddings[0], embeddings[2])
print(f"HP 80A vs Epson Cyan similarity: {sim2:.4f}")

print(f"\nExpected: First similarity > Second similarity")
print(f"Result: {sim1 > sim2} ✓" if sim1 > sim2 else f"Result: {sim1 > sim2} ✗")