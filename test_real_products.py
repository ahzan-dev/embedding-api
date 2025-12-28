import requests

# Your actual product descriptions
products = [
    "Product: 4R RC Gloss Photo Paper 260gsm. Description: Size 4R size (102mm*152mm) Pack 100 sheets Finish Resin coated Gloss photo paper GSM 260 gsm",
    "Product: PVC Ribbon Blank ID Card. Description: Surface Finish: Glossy Material: PVC Color: White",
    "Product: Refill ink Black. Description: 1 bottle of Black ink 100 ml Premium quality"
]

response = requests.post('http://localhost:8000/embed', json={
    'texts': products
})

data = response.json()
print(f"✓ Successfully embedded {data['count']} products")
print(f"✓ Each embedding has {data['dimensions']} dimensions")
print(f"\nFirst 10 values of first embedding:")
print(data['embeddings'][0][:10])