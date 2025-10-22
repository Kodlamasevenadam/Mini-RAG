from sentence_transformers import SentenceTransformer, util

# 1) Küçük kütüphane veri tabanı
library = [
    {"title": "Matematiğin Kısa Tarihi",
        "author": "Ian Stewart", "subject": "Matematik, Tarih"},
    {"title": "Modern Fizik", "author": "Kenneth Krane", "subject": "Fizik"},
    {"title": "Osmanlı Tarihi", "author": "Halil İnalcık", "subject": "Tarih"},
    {"title": "Veri Bilimine Giriş", "author": "Joel Grus",
        "subject": "Bilgisayar, Veri"},
    {"title": "Kuantum Mekaniğine Giriş",
        "author": "David Griffiths", "subject": "Fizik"},
    {"title": "Cumhuriyet Dönemi Türk Edebiyatı",
        "author": "Mehmet Kaplan", "subject": "Edebiyat, Tarih"}
]

# 2) Embedding modeli yükle
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3) Kitap açıklamalarını (title + subject) encode et
library_texts = [book["title"] + " - " + book["subject"] for book in library]
library_embeddings = model.encode(library_texts, convert_to_tensor=True)


def rag_query(user_query, top_k=3):
    # 4) Kullanıcı sorgusunu encode et
    query_embedding = model.encode(user_query, convert_to_tensor=True)

    # 5) En benzer kitapları bul
    hits = util.semantic_search(
        query_embedding, library_embeddings, top_k=top_k)[0]

    # 6) Sonuçları getir
    results = []
    for hit in hits:
        book = library[hit["corpus_id"]]
        results.append((book["title"], book["author"],
                       book["subject"], hit["score"]))

    return results


# Kullanıcıdan sorgu al
query = input("Aramak istediğiniz kitabı veya konuyu yazın: ")

results = rag_query(query)

print(f"\nSorgu: {query}\n")
print("En uygun kitaplar:")
for title, author, subject, score in results:
    print(f"- {title} ({author}) | Konu: {subject} | Benzerlik: {score:.2f}")
