import matplotlib.pyplot as plt
import networkx as nx

# Architecture definition
# Format: [op_type, skip_0, skip_1, ..., skip_i-1]
architecture = [
    [1],
    [3, 1],
    [3, 0, 1],
    [3, 1, 0, 1],
    [4, 1, 0, 0, 0],
    [0, 1, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [3, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [4, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [5, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0],
    [5, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    [3, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1],
]

# Operation mapping
op_map = {
    0: 'conv 3x3',
    1: 'conv 5x5',
    2: 'sep 3x3',
    3: 'sep 5x5',
    4: 'max pool',
    5: 'avg pool'
}

# Colors for each op type
op_colors = {
    'conv 3x3': '#A8DADC',
    'conv 5x5': '#FCD5CE',
    'sep 3x3': '#FFEBA1',
    'sep 5x5': '#BDE0FE',
    'max pool': '#CDB4DB',
    'avg pool': '#D8F3DC'
}
""" This code has been developed by Tamai Ramírez Gordillo (GitHub: TamaiRamirezUA)"""
# Create the graph
G = nx.DiGraph()

# Add input and softmax nodes
G.add_node("input", label="input", color="#a0d2eb")
G.add_node("softmax", label="Softmax", color="#f4cccc")

# Add operation nodes and connections
for i, layer in enumerate(architecture):
    op_type = op_map[layer[0]]
    label = op_type
    node_name = f"L{i}"
    G.add_node(node_name, label=label, color=op_colors[op_type])

    # Always connect to previous node
    if i == 0:
        G.add_edge("input", node_name)
    else:
        for j, skip in enumerate(layer[1:]):
            if skip == 1:
                from_node = f"L{j}" if j >= 0 else "input"
                G.add_edge(from_node, node_name)

# Connect final node to softmax
G.add_edge(f"L{len(architecture) - 1}", "softmax")

# Draw the graph
pos = nx.nx_agraph.graphviz_layout(G, prog='dot', args='-Grankdir=LR')
node_colors = [G.nodes[n].get('color', '#ffffff') for n in G.nodes]
node_labels = {n: G.nodes[n].get('label', n) for n in G.nodes}

plt.figure(figsize=(15, 10))
nx.draw(G, pos, labels=node_labels, node_color=node_colors,
        with_labels=True, node_size=4300, edge_color='black', arrows=True, font_size=14)

plt.title("ENAS Final Architecture")
plt.axis('off')
output_dir = "/workspace/flwr-cifar10-enas/flwr_cifar10_enas"
plt.savefig(f"{output_dir}/fl-enas_architecture.png")
plt.close()

print(f"Gráficos guardados en: {output_dir}/")