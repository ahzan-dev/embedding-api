import requests
import time

# Generate 100 fake products
products = [f"Product {i}: Test ink cartridge black" for i in range(100)]

start = time.time()
response = requests.post('http://localhost:8000/embed', json={
    'texts': products,
    'batch_size': 32
})
elapsed = time.time() - start

data = response.json()
print(f"Embedded {data['count']} products in {elapsed:.2f} seconds")
print(f"Average: {elapsed/data['count']*1000:.2f}ms per product")