import pickle

with open("embeddings.pkl", "rb") as f:
    data = pickle.load(f)

print(type(data))
print(data[:2] if isinstance(data, list) else data)
