"""Flask application for the Fraud Graph Demo.

Serves the interactive fraud graph visualization and REST API endpoints
for searching, filtering, exporting, and importing graph data.
"""

import argparse
import json
import logging
import os
import socket
import sys

from flask import Flask, jsonify, redirect, render_template, request, send_file

from data_loader import generate_synthetic, load_from_file
from fraud_scorer import detect_fraud_rings, score_graph
from graph_engine import FraudGraph
from models import FraudRing
from visualizer import build_pyvis_graph, regenerate_filtered, render_html

# Module-level globals for graph state
fraud_graph: FraudGraph = FraudGraph()
fraud_rings: list[FraudRing] = []

app = Flask(__name__)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Search and filter helper functions (Task 8.2)
# ---------------------------------------------------------------------------


def search_nodes(query: str) -> list[str]:
    """Search nodes by name or identifier (case-insensitive substring match).

    Returns list of matching node IDs.
    """
    if not query:
        return list(fraud_graph.graph.nodes())

    query_lower = query.lower()
    matching = []
    for node_id, data in fraud_graph.graph.nodes(data=True):
        name = data.get("name", "").lower()
        id_lower = node_id.lower()
        if query_lower in name or query_lower in id_lower:
            matching.append(node_id)
    return matching


def filter_by_type(node_ids: list[str], entity_type: str) -> list[str]:
    """Filter node IDs to only those matching the given entity type."""
    if not entity_type:
        return node_ids
    return [
        nid for nid in node_ids
        if fraud_graph.graph.nodes[nid].get("entity_type", "") == entity_type
    ]


def filter_by_tier(node_ids: list[str], tier: str) -> list[str]:
    """Filter node IDs to only those in the given risk tier (low, medium, high)."""
    if not tier:
        return node_ids

    result = []
    for nid in node_ids:
        score = fraud_graph.graph.nodes[nid].get("risk_score", 0.0)
        if tier == "low" and score < 0.3:
            result.append(nid)
        elif tier == "medium" and 0.3 <= score < 0.7:
            result.append(nid)
        elif tier == "high" and score >= 0.7:
            result.append(nid)
    return result


# ---------------------------------------------------------------------------
# Flask routes (Task 8.1)
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    """Serve the main page with sidebar and iframe."""
    return render_template("index.html")


@app.route("/graph")
def graph():
    """Serve the PyVis-generated interactive graph HTML."""
    graph_path = os.path.join("templates", "graph.html")
    if not os.path.exists(graph_path):
        return "Graph not yet generated", 503
    return send_file(graph_path)


@app.route("/api/filter")
def api_filter():
    """Regenerate filtered graph and redirect to /graph.

    Query params:
      - type: entity type filter
      - tier: risk tier filter (low, medium, high)
      - q: search query (name or ID substring, or comma-separated node IDs for ring filter)
    """
    structured_ids = _resolve_from_search_fields()
    if structured_ids:
        _regenerate_for_node_ids(_expand_one_hop(structured_ids))
        return redirect("/graph")

    entity_type = request.args.get("type", "").strip() or None
    risk_tier = request.args.get("tier", "").strip() or None
    search_query = request.args.get("q", "").strip() or None
    search_scope = request.args.get("scope", "").strip() or None

    scope_entity_type = _scope_to_entity_type(search_scope)
    if scope_entity_type and not entity_type:
        entity_type = scope_entity_type

    # Backward-compatible person-name expansion:
    # if only a search query is provided and it matches Person selections,
    # expand to full person affiliations context.
    if search_query and not risk_tier and (not search_scope or search_scope in {"all", "name"}):
        resolved_ids = _resolve_selection_tokens([search_query], search_scope=search_scope)
        has_person = any(
            fraud_graph.graph.has_node(nid) and
            fraud_graph.graph.nodes[nid].get("entity_type", "") == "Person"
            for nid in resolved_ids
        )
        if has_person:
            context_ids = _expand_person_context(resolved_ids)
            if context_ids:
                _regenerate_for_node_ids(context_ids)
                return redirect("/graph")

    # Check if search_query is a comma-separated list of node IDs (ring filter)
    if search_query and "," in search_query:
        candidate_ids = [s.strip() for s in search_query.split(",")]
        # Verify these are actual node IDs
        valid_ids = [nid for nid in candidate_ids if fraud_graph.graph.has_node(nid)]
        if valid_ids:
            # This is a ring filter - regenerate with only these nodes
            _regenerate_for_node_ids(valid_ids)
            return redirect("/graph")

    # Use the visualizer's regenerate_filtered which handles all filtering
    regenerate_filtered(
        fraud_graph,
        fraud_rings,
        entity_type=entity_type,
        risk_tier=risk_tier,
        search_query=search_query,
    )

    return redirect("/graph")


def _regenerate_for_node_ids(node_ids: list[str]):
    """Regenerate the graph visualization showing only the specified node IDs."""
    from models import FraudRing as FR

    matching_set = set(node_ids)
    filtered_graph = FraudGraph()
    filtered_graph.graph = fraud_graph.graph.subgraph(matching_set).copy()

    # Filter fraud rings to only include those with members in the filtered set
    filtered_rings: list[FraudRing] = []
    for ring in fraud_rings:
        filtered_members = [mid for mid in ring.member_ids if mid in matching_set]
        if filtered_members:
            filtered_rings.append(
                FR(
                    ring_id=ring.ring_id,
                    member_ids=filtered_members,
                    avg_risk_score=ring.avg_risk_score,
                )
            )

    network = build_pyvis_graph(filtered_graph, filtered_rings)
    render_html(network)


def _scope_to_entity_type(search_scope: str | None) -> str | None:
    """Map search scope to an entity type filter."""
    mapping = {
        "name": "Person",
        "phone": "Phone_Number",
        "address": "Address",
        "ip": "IP_Address",
        "account": "Account",
    }
    if not search_scope:
        return None
    return mapping.get(search_scope)


def _merge_unique_ids(*id_lists: list[str]) -> list[str]:
    """Merge multiple node ID lists preserving first-seen order."""
    merged: list[str] = []
    seen: set[str] = set()
    for id_list in id_lists:
        for node_id in id_list:
            if node_id not in seen:
                seen.add(node_id)
                merged.append(node_id)
    return merged


def _resolve_selection_tokens(tokens: list[str], search_scope: str | None = None) -> list[str]:
    """Resolve user-provided selection tokens to node IDs.

    Priority:
    1) Direct node ID match.
    2) Exact case-insensitive node name match.
    3) Case-insensitive node name substring match.
    """
    resolved: list[str] = []
    seen: set[str] = set()

    scoped_type = _scope_to_entity_type(search_scope)

    for token in tokens:
        raw_token = token.strip()
        if not raw_token:
            continue

        token_lower_prefix = raw_token.lower()
        if token_lower_prefix.startswith("name:"):
            token = raw_token.split(":", 1)[1].strip()
        elif token_lower_prefix.startswith("id:"):
            token = raw_token.split(":", 1)[1].strip()
        else:
            token = raw_token

        if not token:
            continue

        if fraud_graph.graph.has_node(token):
            if scoped_type and fraud_graph.graph.nodes[token].get("entity_type", "") != scoped_type:
                continue
            if token not in seen:
                resolved.append(token)
                seen.add(token)
            continue

        token_lower = token.lower()
        exact_matches = []
        partial_matches = []

        for node_id, data in fraud_graph.graph.nodes(data=True):
            if scoped_type and data.get("entity_type", "") != scoped_type:
                continue
            name = str(data.get("name", ""))
            name_lower = name.lower()
            if name_lower == token_lower:
                exact_matches.append(node_id)
            elif token_lower in name_lower:
                partial_matches.append(node_id)

        candidates = exact_matches if exact_matches else partial_matches
        for node_id in candidates:
            if node_id not in seen:
                resolved.append(node_id)
                seen.add(node_id)

    return resolved


def _resolve_from_search_fields() -> list[str]:
    """Resolve node IDs from structured search fields in request args."""
    names = request.args.get("name", "").strip()
    phones = request.args.get("phone", "").strip()
    addresses = request.args.get("address", "").strip()
    accounts = request.args.get("account", "").strip()
    ips = request.args.get("ip", "").strip()

    if not any([names, phones, addresses, accounts, ips]):
        return []

    def split_tokens(raw: str) -> list[str]:
        return [s.strip() for s in raw.split(",") if s.strip()]

    return _merge_unique_ids(
        _resolve_selection_tokens(split_tokens(names), search_scope="name"),
        _resolve_selection_tokens(split_tokens(phones), search_scope="phone"),
        _resolve_selection_tokens(split_tokens(addresses), search_scope="address"),
        _resolve_selection_tokens(split_tokens(accounts), search_scope="account"),
        _resolve_selection_tokens(split_tokens(ips), search_scope="ip"),
    )


def _expand_one_hop(node_ids: list[str]) -> list[str]:
    """Return selected nodes plus their direct neighbors (1-degree)."""
    g = fraud_graph.graph
    selected_set: set[str] = {nid for nid in node_ids if g.has_node(nid)}
    one_hop = set(selected_set)
    for nid in selected_set:
        one_hop.update(g.neighbors(nid))
    return sorted(one_hop)


def _expand_investigation_neighborhood(seed_ids: list[str]) -> list[str]:
    """Expand to a broader, but bounded, investigation neighborhood.

    Includes:
    - Anchor Person nodes (or people connected to selected entities)
    - Their profile entities (account/address/phone/ip/device/transaction)
    - People sharing those entities (or same name)
    - Profile entities/transactions for those related people
    """
    g = fraud_graph.graph
    allowed_types = {
        "Person",
        "Account",
        "Transaction",
        "Address",
        "Phone_Number",
        "IP_Address",
        "Device",
    }
    shared_attr_types = {"Address", "Phone_Number", "IP_Address", "Device"}

    # Resolve anchor persons from selected seeds (person or identifier nodes).
    person_ids: set[str] = set()
    seed_entities: set[str] = set()
    for node_id in seed_ids:
        if not g.has_node(node_id):
            continue
        node_type = g.nodes[node_id].get("entity_type", "")
        if node_type == "Person":
            person_ids.add(node_id)
        elif node_type in allowed_types:
            seed_entities.add(node_id)
            for neighbor in g.neighbors(node_id):
                if g.nodes[neighbor].get("entity_type", "") == "Person":
                    person_ids.add(neighbor)
                elif node_type == "Transaction" and g.nodes[neighbor].get("entity_type", "") == "Account":
                    for second_hop in g.neighbors(neighbor):
                        if g.nodes[second_hop].get("entity_type", "") == "Person":
                            person_ids.add(second_hop)

    if not person_ids and not seed_entities:
        return []

    selected_set: set[str] = set(person_ids)
    selected_set.update(seed_entities)

    # Include people who share the same name as selected person(s).
    selected_names = {
        str(g.nodes[pid].get("name", "")).strip().lower()
        for pid in person_ids
        if g.has_node(pid)
    }
    for node_id, data in g.nodes(data=True):
        if data.get("entity_type", "") == "Person":
            name_lower = str(data.get("name", "")).strip().lower()
            if name_lower and name_lower in selected_names:
                selected_set.add(node_id)
                person_ids.add(node_id)

    # First pass from selected people.
    account_ids: set[str] = set()
    shared_entity_ids: set[str] = set()
    transaction_ids: set[str] = set()
    for person_id in list(person_ids):
        for neighbor in g.neighbors(person_id):
            n_type = g.nodes[neighbor].get("entity_type", "")
            if n_type == "Account":
                account_ids.add(neighbor)
            elif n_type == "Transaction":
                transaction_ids.add(neighbor)
            elif n_type in shared_attr_types:
                shared_entity_ids.add(neighbor)

    # Include selected non-person seeds in proper buckets.
    for node_id in seed_entities:
        n_type = g.nodes[node_id].get("entity_type", "")
        if n_type == "Account":
            account_ids.add(node_id)
        elif n_type == "Transaction":
            transaction_ids.add(node_id)
        elif n_type in shared_attr_types:
            shared_entity_ids.add(node_id)

    selected_set.update(account_ids)
    selected_set.update(shared_entity_ids)
    selected_set.update(transaction_ids)

    # Add transactions attached to selected people's accounts.
    account_transaction_ids: set[str] = set()
    for account_id in account_ids:
        for neighbor in g.neighbors(account_id):
            if g.nodes[neighbor].get("entity_type", "") == "Transaction":
                account_transaction_ids.add(neighbor)
    transaction_ids.update(account_transaction_ids)
    selected_set.update(account_transaction_ids)

    # Add other people connected through shared entities/accounts.
    related_people: set[str] = set()
    for entity_id in shared_entity_ids:
        for neighbor in g.neighbors(entity_id):
            if g.nodes[neighbor].get("entity_type", "") == "Person":
                related_people.add(neighbor)
    for account_id in account_ids:
        for neighbor in g.neighbors(account_id):
            if g.nodes[neighbor].get("entity_type", "") == "Person":
                related_people.add(neighbor)

    selected_set.update(related_people)

    # For related people, include key profile entities and transactions.
    related_accounts: set[str] = set()
    related_shared_entities: set[str] = set()
    related_transactions: set[str] = set()
    for person_id in related_people:
        for neighbor in g.neighbors(person_id):
            n_type = g.nodes[neighbor].get("entity_type", "")
            if n_type == "Account":
                related_accounts.add(neighbor)
            elif n_type == "Transaction":
                related_transactions.add(neighbor)
            elif n_type in shared_attr_types:
                related_shared_entities.add(neighbor)
    for account_id in related_accounts:
        for neighbor in g.neighbors(account_id):
            if g.nodes[neighbor].get("entity_type", "") == "Transaction":
                related_transactions.add(neighbor)

    selected_set.update(related_accounts)
    selected_set.update(related_shared_entities)
    selected_set.update(related_transactions)

    return sorted(selected_set)


def _expand_person_context(seed_ids: list[str]) -> list[str]:
    """Expand to the full connected investigation component(s)."""
    g = fraud_graph.graph
    seeds = [nid for nid in seed_ids if g.has_node(nid)]
    if not seeds:
        return []

    visited: set[str] = set(seeds)
    frontier: set[str] = set(seeds)
    while frontier:
        next_frontier: set[str] = set()
        for nid in frontier:
            for neighbor in g.neighbors(nid):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
        frontier = next_frontier

    return sorted(visited)


@app.route("/api/select")
def api_select():
    """Show only selected nodes (and optionally their neighbors).

    Query params:
      - ids: comma-separated node IDs
      - neighbors: if "1", include direct neighbors of selected nodes
    """
    ids_raw = request.args.get("ids", "").strip()
    valid_ids: list[str] = []

    if ids_raw:
        node_tokens = [s.strip() for s in ids_raw.split(",") if s.strip()]
        search_scope = request.args.get("scope", "").strip() or None
        valid_ids = _resolve_selection_tokens(node_tokens, search_scope=search_scope)

    structured_ids = _resolve_from_search_fields()
    if structured_ids:
        valid_ids = _merge_unique_ids(valid_ids, structured_ids)

    if not valid_ids:
        return jsonify({"error": "No matching nodes found for provided search criteria"}), 404

    person_context = request.args.get("person_context", "") == "1"
    has_person_selection = any(
        fraud_graph.graph.nodes[nid].get("entity_type", "") == "Person"
        for nid in valid_ids
        if fraud_graph.graph.has_node(nid)
    )
    context_mode = request.args.get("context_mode", "").strip() or "neighborhood"

    # Auto-expand for Person selections so name-based selection always shows affiliations.
    if person_context or has_person_selection:
        if context_mode == "full":
            context_ids = _expand_person_context(valid_ids)
        elif context_mode == "neighborhood":
            context_ids = _expand_investigation_neighborhood(valid_ids)
        else:
            context_ids = _expand_one_hop(valid_ids)
        if not context_ids:
            return jsonify({"error": "Person context mode requires at least one valid Person selection"}), 400
        _regenerate_for_node_ids(context_ids)
        return redirect("/graph")

    include_neighbors = request.args.get("neighbors", "") == "1"

    # Expand to include neighbors if requested
    selected_set = set(valid_ids)
    if include_neighbors:
        for nid in valid_ids:
            selected_set.update(fraud_graph.get_neighbors(nid))

    _regenerate_for_node_ids(list(selected_set))
    return redirect("/graph")


@app.route("/api/node/<path:node_id>")
def api_node(node_id: str):
    """Return single node details as JSON."""
    try:
        node_data = fraud_graph.get_node(node_id)
        node_data["id"] = node_id
        # Get neighbors
        neighbors = fraud_graph.get_neighbors(node_id)
        node_data["neighbors"] = neighbors
        return jsonify(node_data)
    except KeyError:
        return jsonify({"error": f"Node '{node_id}' not found"}), 404


@app.route("/api/rings")
def api_rings():
    """Return list of fraud rings as JSON."""
    rings_data = []
    for ring in fraud_rings:
        rings_data.append({
            "ring_id": ring.ring_id,
            "member_ids": ring.member_ids,
            "member_count": len(ring.member_ids),
            "avg_risk_score": round(ring.avg_risk_score, 3),
        })
    return jsonify(rings_data)


@app.route("/api/export")
def api_export():
    """Download graph as JSON file."""
    export_data = fraud_graph.export_json()
    # Write to a temp file and send
    export_path = "fraud_graph_export.json"
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2)
    return send_file(
        export_path,
        mimetype="application/json",
        as_attachment=True,
        download_name="fraud_graph_export.json",
    )


@app.route("/api/import", methods=["POST"])
def api_import():
    """Upload JSON file to reconstruct graph."""
    global fraud_graph, fraud_rings

    if "file" not in request.files:
        return jsonify({"error": "No file provided. Use 'file' field."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        data = json.load(file)
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Invalid JSON: {e}"}), 400

    try:
        fraud_graph.import_json(data)
        fraud_rings = fraud_graph.fraud_rings

        # Regenerate visualization
        network = build_pyvis_graph(fraud_graph, fraud_rings)
        render_html(network)

        return jsonify({"message": "Graph imported successfully", "node_count": fraud_graph.graph.number_of_nodes()})
    except (ImportError, Exception) as e:
        return jsonify({"error": f"Import failed: {e}"}), 400


# ---------------------------------------------------------------------------
# App startup (Task 8.6)
# ---------------------------------------------------------------------------


def is_port_available(port: int) -> bool:
    """Check if a port is available for binding."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def find_available_port(start_port: int = 5000, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port."""
    for offset in range(max_attempts):
        port = start_port + offset
        if is_port_available(port):
            return port
    return start_port


def main():
    """CLI entry point: parse args, load data, build graph, score, visualize, serve."""
    global fraud_graph, fraud_rings

    # Check dependencies
    missing_deps = []
    try:
        import flask  # noqa: F401
    except ImportError:
        missing_deps.append("flask")
    try:
        import networkx  # noqa: F401
    except ImportError:
        missing_deps.append("networkx")
    try:
        import pyvis  # noqa: F401
    except ImportError:
        missing_deps.append("pyvis")

    if missing_deps:
        for dep in missing_deps:
            print(f"Missing dependency: {dep}. Install with: pip3 install {dep}")
        print(f"\nOr install all at once: pip3 install {' '.join(missing_deps)}")
        sys.exit(1)

    # Parse CLI arguments
    parser = argparse.ArgumentParser(
        description="Fraud Graph Demo - Interactive fraud graph visualization"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to CSV or JSON data file. If not provided, synthetic data is generated.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic data generation (default: 42)",
    )
    parser.add_argument(
        "--entities",
        type=int,
        default=200,
        help="Number of entities for synthetic data (default: 200)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to run the Flask server on (default: 5000)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run Flask in debug mode",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Step 1: Load data
    print("=" * 60)
    print("  Fraud Graph Demo")
    print("=" * 60)

    if args.data:
        print(f"\n[1/5] Loading data from: {args.data}")
        try:
            entities, relationships = load_from_file(args.data)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)
    else:
        print(f"\n[1/5] Generating synthetic data (seed={args.seed}, entities={args.entities})")
        entities, relationships = generate_synthetic(seed=args.seed, num_entities=args.entities)

    print(f"      Loaded {len(entities)} entities and {len(relationships)} relationships")

    # Step 2: Build graph
    print("[2/5] Building graph...")
    fraud_graph = FraudGraph()
    fraud_graph.build(entities, relationships)
    print(f"      Graph: {fraud_graph.graph.number_of_nodes()} nodes, {fraud_graph.graph.number_of_edges()} edges")

    # Step 3: Score entities and detect fraud rings
    print("[3/5] Scoring entities and detecting fraud rings...")
    scores = score_graph(fraud_graph)
    fraud_rings = fraud_graph.fraud_rings
    print(f"      Detected {len(fraud_rings)} fraud rings")

    from fraud_scorer import summarize_risk_tiers
    tiers = summarize_risk_tiers(scores)
    print(f"      Risk tiers: {tiers['high']} high, {tiers['medium']} medium, {tiers['low']} low")

    # Step 4: Generate PyVis visualization
    print("[4/5] Generating interactive visualization...")
    network = build_pyvis_graph(fraud_graph, fraud_rings)
    render_html(network)
    print("      Saved to templates/graph.html")

    # Step 5: Start Flask server
    port = args.port
    if not is_port_available(port):
        new_port = find_available_port(port + 1)
        if new_port != port + 1 or not is_port_available(new_port):
            print(f"\n[WARN] Port {port} is in use. Try: python3 app.py --port {port + 1}")
            print(f"   Or kill the process using port {port} and retry.")
            sys.exit(1)
        port = new_port
        print(f"      Port {args.port} in use, using port {port} instead")

    print(f"\n[5/5] Starting Flask server...")
    print(f"\n{'=' * 60}")
    print(f"  Open in browser: http://127.0.0.1:{port}")
    print(f"{'=' * 60}\n")

    app.run(host="127.0.0.1", port=port, debug=args.debug)


if __name__ == "__main__":
    main()
