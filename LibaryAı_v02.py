import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Database oku ve kolonları genelleştir
df = pd.read_csv("kutuphane.csv")  # kendi dosya yolunu yaz
df.columns = df.columns.str.strip()  # baş/son boşlukları temizle

# Kolonları standart isimlere çevir (birden fazla olası yazımı destekle)
rename_map = {}
for c in df.columns:
    k = c.strip().lower()
    if k in {"catagory", "category", "kategori", "subject", "topic"}:
        rename_map[c] = "category"
    elif k in {"title", "kitap", "kitap adı", "book", "book_title", "name"}:
        rename_map[c] = "title"
    elif k in {"author", "yazar", "writer"}:
        rename_map[c] = "author"
    elif k in {"status", "durum", "availability", "available", "state"}:
        rename_map[c] = "status"
    elif k in {"bid", "id", "book_id"}:
        rename_map[c] = "bid"

df = df.rename(columns=rename_map)

#  kolonları kontrol et
required = ["title", "category"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(
        f"Gerekli kolon(lar) eksik: {missing}. Mevcut kolonlar: {list(df.columns)}")

# Boşları güvenli hale getir
for col in ["title", "author", "category", "status", "bid"]:
    if col not in df.columns:
        df[col] = ""
df[["title", "author", "category", "status", "bid"]] = df[[
    "title", "author", "category", "status", "bid"]].fillna("")

# --- 2) Kütüphane kayıtlarını hazırla
library = df.to_dict(orient="records")

# --- 3) Embedding modeli
model = SentenceTransformer("all-MiniLM-L6-v2")

# Arama metnini title+category üzerinden kur
library_texts = [
    f"{b.get('title', '')} - {b.get('category', '')}" for b in library]
library_embeddings = model.encode(library_texts, convert_to_tensor=True)

# --- 4) RAG sorgu fonksiyonu


def rag_query(user_query, top_k=3):
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    hits = util.semantic_search(
        query_embedding, library_embeddings, top_k=top_k)[0]

    results = []
    for h in hits:
        b = library[h["corpus_id"]]
        results.append({
            "bid": b.get("bid", ""),
            "title": b.get("title", ""),
            "author": b.get("author", ""),
            "category": b.get("category", ""),
            "status": b.get("status", ""),
            "score": float(h["score"]),
        })
    return results


# --- 5) CLI
query = input("Aramak istediğiniz kitabı veya konuyu yazın: ")
results = rag_query(query, top_k=5)

print(f"\nSorgu: {query}\n")
print("En uygun kitaplar:")
for r in results:
    bid_prefix = f"[{r['bid']}] " if str(r['bid']).strip() else ""
    print(f"{bid_prefix}{r['title']} ({r['author']}) | Kategori: {r['category']} | "
          f"Durum: {r['status']} | Benzerlik: {r['score']:.2f}")
