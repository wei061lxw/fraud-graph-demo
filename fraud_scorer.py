"""Fraud Scorer for the Fraud Graph Demo.

Computes risk scores for entities based on graph topology (degree centrality,
shared attributes) and detects fraud rings using connected component analysis.
"""

import uuid
from collections import defaultdict

import networkx as nx

from graph_engine import FraudGraph
from models import FraudRing

# Attribute entity types that indicate shared resources
ATTRIBUTE_TYPES = {"Address", "IP_Address", "Device", "Phone_Number"}


def detect_fraud_rings(graph: FraudGraph) -> list[FraudRing]:
    """Detect clusters of entities sharing multiple attributes.

    Uses connected component analysis on a subgraph where Person/Account
    entities are connected if they share 2 or more attribute nodes
    (Address, IP_Address, Device, Phone_Number).

    Returns a list of FraudRing objects with member IDs and average risk score.
    """
    g = graph.graph

    if g.number_of_nodes() == 0:
        return []

    # Identify attribute nodes (Address, IP_Address, Device, Phone_Number)
    attribute_nodes = set()
    non_attribute_nodes = set()
    for node_id, data in g.nodes(data=True):
        entity_type = data.get("entity_type", "")
        if entity_type in ATTRIBUTE_TYPES:
            attribute_nodes.add(node_id)
        else:
            non_attribute_nodes.add(node_id)

    if not attribute_nodes or not non_attribute_nodes:
        return []

    # For each attribute node, find which non-attribute entities connect to it
    # Then build a mapping: for each pair of non-attribute entities,
    # count how many attribute nodes they share
    attr_to_entities: dict[str, set[str]] = defaultdict(set)
    for attr_node in attribute_nodes:
        if g.has_node(attr_node):
            for neighbor in g.neighbors(attr_node):
                if neighbor in non_attribute_nodes:
                    attr_to_entities[attr_node].add(neighbor)

    # Build a shared-attribute subgraph: connect two non-attribute entities
    # if they share >= 2 attribute nodes
    shared_count: dict[tuple[str, str], int] = defaultdict(int)
    for attr_node, connected_entities in attr_to_entities.items():
        entity_list = sorted(connected_entities)
        for i in range(len(entity_list)):
            for j in range(i + 1, len(entity_list)):
                pair = (entity_list[i], entity_list[j])
                shared_count[pair] += 1

    # Build subgraph with edges where shared count >= 2
    shared_subgraph = nx.Graph()
    for (e1, e2), count in shared_count.items():
        if count >= 2:
            shared_subgraph.add_edge(e1, e2)

    if shared_subgraph.number_of_nodes() == 0:
        return []

    # Find connected components - each component is a potential fraud ring
    # Only consider components with 2+ members
    fraud_rings: list[FraudRing] = []
    for component in nx.connected_components(shared_subgraph):
        if len(component) >= 2:
            member_ids = sorted(component)
            ring_id = f"ring_{uuid.uuid4().hex[:8]}"
            fraud_rings.append(FraudRing(
                ring_id=ring_id,
                member_ids=member_ids,
                avg_risk_score=0.0,  # Will be updated after scoring
            ))

    return fraud_rings


def score_graph(graph: FraudGraph) -> dict[str, float]:
    """Compute risk scores for all entities. Returns {node_id: score}.

    Scoring factors:
    - Degree centrality (normalized)
    - Shared attribute count with neighbors
    - Membership in detected fraud rings (>= 0.7 if in ring)

    All scores are clamped to [0.0, 1.0].
    """
    g = graph.graph

    if g.number_of_nodes() == 0:
        return {}

    # Step 1: Detect fraud rings
    fraud_rings = detect_fraud_rings(graph)
    ring_members: set[str] = set()
    for ring in fraud_rings:
        ring_members.update(ring.member_ids)

    # Step 2: Compute degree centrality (normalized to [0, 1])
    degree_centrality = nx.degree_centrality(g)

    # Step 3: Compute shared attribute score for each node
    # For each non-attribute node, count how many attribute-type neighbors it has
    attribute_nodes = set()
    for node_id, data in g.nodes(data=True):
        if data.get("entity_type", "") in ATTRIBUTE_TYPES:
            attribute_nodes.add(node_id)

    # Find the max shared attribute count for normalization
    shared_attr_counts: dict[str, int] = {}
    for node_id in g.nodes():
        if node_id in attribute_nodes:
            shared_attr_counts[node_id] = 0
            continue
        count = sum(1 for neighbor in g.neighbors(node_id) if neighbor in attribute_nodes)
        shared_attr_counts[node_id] = count

    max_shared = max(shared_attr_counts.values()) if shared_attr_counts else 1
    if max_shared == 0:
        max_shared = 1

    # Step 4: Compute base scores
    scores: dict[str, float] = {}
    for node_id in g.nodes():
        dc = degree_centrality.get(node_id, 0.0)
        shared_norm = shared_attr_counts.get(node_id, 0) / max_shared

        # Weighted combination: 50% degree centrality, 50% shared attributes
        base_score = 0.5 * dc + 0.5 * shared_norm

        # Boost fraud ring members to at least 0.7
        if node_id in ring_members:
            base_score = max(base_score, 0.7)

        # Clamp to [0.0, 1.0]
        scores[node_id] = max(0.0, min(1.0, base_score))

    # Step 5: Update fraud ring average scores
    for ring in fraud_rings:
        if ring.member_ids:
            ring.avg_risk_score = sum(
                scores.get(mid, 0.0) for mid in ring.member_ids
            ) / len(ring.member_ids)

    # Step 6: Store scores on graph nodes and update graph's fraud_rings
    for node_id, score in scores.items():
        g.nodes[node_id]["risk_score"] = score

    graph.fraud_rings = fraud_rings

    return scores


def summarize_risk_tiers(scores: dict[str, float]) -> dict[str, int]:
    """Return counts: {'low': n, 'medium': n, 'high': n}.

    Tiers:
    - low: [0.0, 0.3)
    - medium: [0.3, 0.7)
    - high: [0.7, 1.0]
    """
    tiers = {"low": 0, "medium": 0, "high": 0}

    for score in scores.values():
        if score < 0.3:
            tiers["low"] += 1
        elif score < 0.7:
            tiers["medium"] += 1
        else:
            tiers["high"] += 1

    return tiers
