"""PyVis Visualizer for the Fraud Graph Demo.

Converts a scored FraudGraph into an interactive PyVis Network visualization
with color-coded nodes, risk-based sizing, tooltips, and fraud ring highlighting.
"""

import os
from typing import Optional

from pyvis.network import Network

from graph_engine import FraudGraph
from models import FraudRing

# Color map for entity types
ENTITY_COLOR_MAP: dict[str, str] = {
    "Person": "#4e79a7",
    "Account": "#f28e2b",
    "Transaction": "#e15759",
    "Address": "#76b7b2",
    "Phone_Number": "#59a14f",
    "IP_Address": "#edc948",
    "Device": "#b07aa1",
}

# Default color for unknown entity types
DEFAULT_NODE_COLOR = "#999999"

# Fraud ring border styling
FRAUD_RING_BORDER_COLOR = "#ff0000"
FRAUD_RING_BORDER_WIDTH = 3

# Node size range (min and max pixel sizes)
MIN_NODE_SIZE = 10
MAX_NODE_SIZE = 40


def build_pyvis_graph(fraud_graph: FraudGraph, fraud_rings: list[FraudRing]) -> Network:
    """Convert a scored FraudGraph into a PyVis Network object.

    - Color-codes nodes by entity type using a consistent color map
    - Sizes nodes by risk score (higher risk = larger)
    - Adds hover tooltips with entity properties and risk score
    - Highlights fraud ring members with distinct border color (red)
    - Labels edges with relationship type
    - Configures force-directed layout (Barnes-Hut physics)
    - Enables zoom/pan and click-to-highlight-neighbors
    """
    net = Network(
        height="750px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#000000",
        directed=False,
    )

    # Configure Barnes-Hut physics for force-directed layout
    net.barnes_hut(
        gravity=-3000,
        central_gravity=0.3,
        spring_length=100,
        spring_strength=0.04,
        damping=0.09,
    )

    # Enable click-to-highlight-neighbors
    net.set_options("""{
        "interaction": {
            "hover": true,
            "navigationButtons": true,
            "zoomView": true
        },
        "nodes": {
            "borderWidth": 1,
            "borderWidthSelected": 2
        }
    }""")

    # Collect fraud ring member IDs for highlighting
    ring_member_ids: set[str] = set()
    for ring in fraud_rings:
        ring_member_ids.update(ring.member_ids)

    # Add nodes
    g = fraud_graph.graph
    for node_id, data in g.nodes(data=True):
        entity_type = data.get("entity_type", "")
        name = data.get("name", node_id)
        properties = data.get("properties", {})
        risk_score = data.get("risk_score", 0.0)

        # Color by entity type
        color = ENTITY_COLOR_MAP.get(entity_type, DEFAULT_NODE_COLOR)

        # Size proportional to risk score
        size = MIN_NODE_SIZE + (MAX_NODE_SIZE - MIN_NODE_SIZE) * risk_score

        # Build HTML tooltip
        tooltip = _build_tooltip(node_id, entity_type, name, properties, risk_score)

        # Node color configuration
        node_color: dict | str
        if node_id in ring_member_ids:
            # Fraud ring members get a red border
            node_color = {
                "background": color,
                "border": FRAUD_RING_BORDER_COLOR,
            }
            border_width = FRAUD_RING_BORDER_WIDTH
        else:
            node_color = color
            border_width = 1

        net.add_node(
            node_id,
            label=name,
            title=tooltip,
            color=node_color,
            size=size,
            borderWidth=border_width,
        )

    # Add edges with relationship type labels
    for source, target, data in g.edges(data=True):
        rel_type = data.get("rel_type", "")
        net.add_edge(source, target, label=rel_type)

    return net


def _build_tooltip(
    node_id: str,
    entity_type: str,
    name: str,
    properties: dict,
    risk_score: float,
) -> str:
    """Build an HTML tooltip string for a node."""
    lines = [
        f"<b>{name}</b>",
        f"<br><b>ID:</b> {node_id}",
        f"<br><b>Type:</b> {entity_type}",
        f"<br><b>Risk Score:</b> {risk_score:.2f}",
    ]

    if properties:
        lines.append("<br><b>Properties:</b>")
        for key, value in properties.items():
            lines.append(f"<br>&nbsp;&nbsp;{key}: {value}")

    return "".join(lines)


def render_html(network: Network, output_path: str = "templates/graph.html") -> str:
    """Generate the interactive HTML file from the PyVis Network.

    Returns the path to the generated HTML file.
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    network.save_graph(output_path)
    return output_path


def regenerate_filtered(
    fraud_graph: FraudGraph,
    fraud_rings: list[FraudRing],
    entity_type: Optional[str] = None,
    risk_tier: Optional[str] = None,
    search_query: Optional[str] = None,
) -> str:
    """Regenerate the PyVis HTML with filters applied.

    Filters:
    - entity_type: only include nodes of this type
    - risk_tier: only include nodes in this risk tier (low, medium, high)
    - search_query: only include nodes whose name or ID contains this string (case-insensitive)

    Returns path to the regenerated HTML file.
    """
    g = fraud_graph.graph

    # Determine which nodes pass the filter
    matching_node_ids: set[str] = set()

    for node_id, data in g.nodes(data=True):
        # Apply entity_type filter
        if entity_type is not None:
            if data.get("entity_type", "") != entity_type:
                continue

        # Apply risk_tier filter
        if risk_tier is not None:
            risk_score = data.get("risk_score", 0.0)
            if not _matches_risk_tier(risk_score, risk_tier):
                continue

        # Apply search_query filter
        if search_query is not None:
            query_lower = search_query.lower()
            name = data.get("name", "").lower()
            id_lower = node_id.lower()
            if query_lower not in name and query_lower not in id_lower:
                continue

        matching_node_ids.add(node_id)

    # Build a filtered FraudGraph
    filtered_graph = FraudGraph()
    filtered_graph.graph = g.subgraph(matching_node_ids).copy()

    # Filter fraud rings to only include those with members in the filtered set
    filtered_rings: list[FraudRing] = []
    for ring in fraud_rings:
        filtered_members = [mid for mid in ring.member_ids if mid in matching_node_ids]
        if filtered_members:
            filtered_rings.append(
                FraudRing(
                    ring_id=ring.ring_id,
                    member_ids=filtered_members,
                    avg_risk_score=ring.avg_risk_score,
                )
            )

    # Build and render the filtered PyVis graph
    network = build_pyvis_graph(filtered_graph, filtered_rings)
    return render_html(network)


def _matches_risk_tier(risk_score: float, tier: str) -> bool:
    """Check if a risk score falls within the specified tier.

    Tiers:
    - low: [0.0, 0.3)
    - medium: [0.3, 0.7)
    - high: [0.7, 1.0]
    """
    if tier == "low":
        return risk_score < 0.3
    elif tier == "medium":
        return 0.3 <= risk_score < 0.7
    elif tier == "high":
        return risk_score >= 0.7
    return False
