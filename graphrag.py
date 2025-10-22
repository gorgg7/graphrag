
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import numpy as np
import os
from openai import OpenAI

# print(" Loading model (all-MiniLM-L6-v2)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
G = nx.Graph()

manual_folder = "manuals"

if not os.path.exists(manual_folder):
    os.makedirs(manual_folder)
    print(f"📁 Folder '{manual_folder}' created! Put your manuals (.txt) inside it then rerun the script.")
    exit()

print(f"📚 Loading manuals from '{manual_folder}'...")

for filename in os.listdir(manual_folder):
    if filename.endswith(".txt"):
        manual_name = filename.replace("_manual.txt", "")
        path = os.path.join(manual_folder, filename)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for i, para in enumerate(paragraphs):
            node_id = f"{manual_name}_{i}"
            G.add_node(node_id, text=para, manual=manual_name)

# print(" Creating embeddings...")
for node in G.nodes():
    G.nodes[node]["embedding"] = model.encode(G.nodes[node]["text"])

# print("🔗 Building graph edges...")
nodes = list(G.nodes())
for i in range(len(nodes)):
    for j in range(i + 1, len(nodes)):
        emb_i = G.nodes[nodes[i]]["embedding"]
        emb_j = G.nodes[nodes[j]]["embedding"]
        sim = cosine_similarity([emb_i], [emb_j])[0][0]
        if sim > 0.65:
            G.add_edge(nodes[i], nodes[j], weight=sim)

print(f" Graph built with {len(G.nodes())} nodes and {len(G.edges())} edges.")


def retrieve_context(query, top_k=3):
    q_vec = model.encode(query)
    sims = {}

    for node in G.nodes():
        emb = G.nodes[node]["embedding"]
        sims[node] = cosine_similarity([q_vec], [emb])[0][0]

    sorted_nodes = sorted(sims.items(), key=lambda x: x[1], reverse=True)[:top_k]

    context_nodes = set()
    for node, _ in sorted_nodes:
        context_nodes.add(node)
        context_nodes.update(G.neighbors(node))

    manuals = [G.nodes[n]["manual"] for n in context_nodes]
    main_manual = max(set(manuals), key=manuals.count)

    context_text = " ".join(G.nodes[n]["text"] for n in context_nodes if G.nodes[n]["manual"] == main_manual)

    return main_manual, context_text


client = OpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1"
)


def ask(query):
    manual, context = retrieve_context(query)

    prompt = f"""
You are a support assistant for machine manuals.
Use the following context from the **{manual}** manual to answer clearly and briefly.

Context:
{context}

Question: {query}
Answer:
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    print("\n===============================")
    print(f" Manual selected: {manual}")
    print(" Answer:", response.choices[0].message.content)
    print("===============================\n")


if __name__ == "__main__":
    print(" Graph RAG Chatbot ready!")
    print("Type your question (or 'exit' to quit)\n")

    while True:
        query = input(" You: ").strip()
        if query.lower() in ["exit", "quit", "q"]:
            print(" Exiting chatbot.")
            break
        ask(query)
 



















 