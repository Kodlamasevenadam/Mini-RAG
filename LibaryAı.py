from sentence_transformers import SentenceTransformer, util


model = SentenceTransformer('all-MiniLM-L6-v2')

# Veri tabanı
documents = 'Bookset.csv'

# Veri tabanını vektöre dönüştür
doc_embeddings = model.encode(documents, convert_to_tensor=True)

#  Kullanıcıdan sorgu
query = input("Bulmak İstediğiniz Kitap: ")
query_embedding = model.encode(query, convert_to_tensor=True)

# Benzerlikleri hesapla (cosine similarity)
similarities = util.pytorch_cos_sim(query_embedding, doc_embeddings)

# En yakın eşleşmeyi bul
best_match_idx = similarities.argmax()

print("Sorgu:", query)
print("En alakalı sonuç:", documents[best_match_idx])
print("Benzerlik skoru:", similarities[0][best_match_idx].item())
