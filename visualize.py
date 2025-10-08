# # visualize.py
# import networkx as nx
# import matplotlib.pyplot as plt

# def visualize_relations(relations):
#     """
#     Visualize extracted entity relationships as a graph.
#     :param relations: List of dicts with keys: entity1, relation, entity2
#     """
#     G = nx.DiGraph()

#     for r in relations:
#         G.add_edge(r["entity1"], r["entity2"], label=r["relation"])

#     pos = nx.spring_layout(G, k=1.2)
#     plt.figure(figsize=(10, 6))
#     nx.draw(G, pos, with_labels=True, node_color="skyblue",
#             node_size=2000, arrowsize=20, font_size=10, font_weight="bold")
#     labels = nx.get_edge_attributes(G, "label")
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color="red", font_size=9)
#     plt.title("Extracted Entity Relations (BERT Fine-tuning Demo)")
#     plt.show()

# if __name__ == "__main__":
#     # Example data (you can replace with real predictions later)
#     example_relations = [
#         {"entity1": "John Doe", "relation": "used_weapon", "entity2": "rifle"},
#         {"entity1": "incident", "relation": "occurred_in", "entity2": "Texas"},
#         {"entity1": "Police", "relation": "handled_by", "entity2": "suspect"}
#     ]
#     visualize_relations(example_relations)
# visualize.py
import json
import networkx as nx
import matplotlib.pyplot as plt

def visualize_relations(relations):
    """Draw graph from a list of relation dicts."""
    G = nx.DiGraph()
    for r in relations:
        G.add_edge(r["entity1"], r["entity2"], label=r["relation"])

    pos = nx.spring_layout(G, k=1.2)
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_color="skyblue",
            node_size=2000, arrowsize=20, font_size=10, font_weight="bold")
    labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color="red", font_size=9)
    plt.title("Extracted Entity Relations (BERT Fine-tuning Demo)")
    plt.show()

def visualize_relations_from_file(json_path="relations.json"):
    """Load relations from JSON file and visualize them."""
    try:
        with open(json_path, "r") as f:
            relations = json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {json_path}")
        return
    if not relations:
        print("⚠️ No relations to visualize (empty file).")
        return
    print(f"✅ Loaded {len(relations)} relations from {json_path}")
    visualize_relations(relations)

if __name__ == "__main__":
    visualize_relations_from_file("relations.json")
