import osmnx as ox
import networkx as nx
import random
import matplotlib.pyplot as plt

# point near the center of dwarka with 3 km radius.
location_point = (28.600503, 77.046880)  # (lat, lon) from market to school
dist_meters = 3000
G = ox.graph_from_point(location_point, dist=dist_meters, network_type='all', simplify=False)

# Print graph size
print(f"Number of nodes: {len(G.nodes())}")
print(f"Number of edges: {len(G.edges())}")
if len(G.edges()) == 0:
    raise ValueError("no roads found")


def filter_cycling_roads(G):
    """
    Keeping edges that are likely suitable for cycling.
    not completely remove main roads (primary, secondary, trunk);
    they will be penalized instead
    """
    G_filtered = G.copy()
    for u, v, k, data in G.edges(keys=True, data=True):
        highway_type = data.get("highway", "")
        if isinstance(highway_type, list):
            highway_type = highway_type[0]
        # cycling friendly types
        if highway_type in ["service", "residential", "footway", "path", "tertiary", "unclassified", "cycleway"]:
            continue  # keep as is
        # For main roads, assign a penalty but not removing completely
        elif highway_type in ["primary", "secondary", "trunk"]:
            data["highway_penalty"] = 2.5  # penalty value / more = higher penalty
        else:
            G_filtered.remove_edge(u, v, key=k)
    return G_filtered


G = filter_cycling_roads(G)
print(f"After filtering - Nodes: {len(G.nodes())}, Edges: {len(G.edges())}")

# =============================================================================

def assign_dynamic_weights(G):
    """
    weight to each edge based on its length and type.
    A small random chance (1%) may block an edge.
    """
    for u, v, k, data in G.edges(keys=True, data=True):
        highway_type = data.get("highway", "")
        if isinstance(highway_type, list):
            highway_type = highway_type[0]
        # Base weight factors.
        if highway_type in ["service", "cycleway"]:
            weight_factor = 0.5
        elif highway_type in ["residential", "unclassified"]:
            weight_factor = 1
        elif highway_type == "tertiary":
            weight_factor = 1.5
        elif highway_type in ["primary", "secondary", "trunk"]:
            weight_factor = data.get("highway_penalty", 2.5)
        else:
            weight_factor = 5
        # simulating a blocked road randomly to 1% chance
        if random.random() < 0.01:
            weight_factor = float('inf')
        data["weight"] = data.get("length", 1) * weight_factor

assign_dynamic_weights(G)


# OSMnx's nearest_nodes() expects (lon, lat)
origin = ox.distance.nearest_nodes(G, 77.046880, 28.600503)
destination = ox.distance.nearest_nodes(G, 77.039215, 28.589114)

if not nx.has_path(G, origin, destination):
    print("origin and destination are disconnected. Using largest scc")
    largest_cc = max(nx.strongly_connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    origin = ox.distance.nearest_nodes(G, 77.046880, 28.600503)
    destination = ox.distance.nearest_nodes(G, 77.039215, 28.589114)
    if not nx.has_path(G, origin, destination):
        raise ValueError("no connected path found between origin and destination.")


def online_planning(G, origin, destination, rollouts=5):
    current_node = origin
    path = [current_node]

    while current_node != destination:
        neighbors = list(G.neighbors(current_node))
        if not neighbors:
            print(f"Dead-end at {current_node} so using global shortest path fallback.")
            try:
                sp = nx.shortest_path(G, source=current_node, target=destination, weight="weight")
                path += sp[1:]
                break
            except nx.NetworkXNoPath:
                print("Global fallback as no path found from current node to destination")
                return []
        best_choice = None
        best_score = float('inf')
        for neighbor in neighbors:
            simulated_costs = []
            assign_dynamic_weights(G)
            for _ in range(rollouts):
                try:
                    cost = nx.shortest_path_length(G, neighbor, destination, weight="weight")
                    simulated_costs.append(cost)
                except nx.NetworkXNoPath:
                    simulated_costs.append(float('inf'))
            avg_cost = sum(simulated_costs) / rollouts
            if avg_cost < best_score:
                best_score = avg_cost
                best_choice = neighbor
        if best_choice is None or best_choice in path:
            print("using global shortest path fallback (loop or no valid neighbors)")
            try:
                sp = nx.shortest_path(G, source=current_node, target=destination, weight="weight")
                path += sp[1:]
                break
            except nx.NetworkXNoPath:
                print("global fallback, no path found.")
                return []
        path.append(best_choice)
        current_node = best_choice
    return path

optimized_path = online_planning(G, origin, destination, rollouts=5)

# print computed path information. (debug)
print(f"Computed Path Length: {len(optimized_path)}")
if optimized_path:
    print("First 10 nodes in path:", optimized_path[:10])
    coords = [(node, G.nodes[node]['x'], G.nodes[node]['y']) for node in optimized_path[:5]]
    print("First 5 node coordinates:", coords)
else:
    print("No valid path computed.")


fig, ax = plt.subplots(figsize=(10,10))

# Plot all edges manually
for u, v, data in G.edges(data=True):
    x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
    x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
    ax.plot([x1, x2], [y1, y2], color='gray', linewidth=0.5, alpha=0.7)

# plot computed routes if any
if optimized_path:
    route_x = [G.nodes[node]['x'] for node in optimized_path]
    route_y = [G.nodes[node]['y'] for node in optimized_path]
    ax.plot(route_x, route_y, color='blue', linewidth=4, zorder=5)
else:
    print("⚠️ No valid route to plot; only the base graph will be shown.")

# Plot start and destination markers (green and red markets at top)
ax.scatter(G.nodes[origin]['x'], G.nodes[origin]['y'], color='red', s=100, zorder=6, label='Start')
ax.scatter(G.nodes[destination]['x'], G.nodes[destination]['y'], color='green', s=100, zorder=6, label='Destination')

# Force the axis limits to the bounding box of all graph nodes.
x_values = [data['x'] for _, data in G.nodes(data=True)]
y_values = [data['y'] for _, data in G.nodes(data=True)]
xmin, xmax = min(x_values), max(x_values)
ymin, ymax = min(y_values), max(y_values)
print("xlim:", xmin, xmax)
print("ylim:", ymin, ymax)
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_aspect('equal', adjustable='datalim')

plt.legend()
plt.title("Cycling Route from Origin to Destination")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

# file save externally
plt.savefig("manual_route_output.png", bbox_inches='tight', dpi=150)
print("saved as 'route_output.png'.")
plt.show()
