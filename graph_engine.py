"""Graph Engine for the Fraud Graph Demo.

Manages the in-memory NetworkX graph, providing methods for building,
querying, exporting, and importing the fraud graph.
"""

import logging
from datetime import datetime, timezone

import networkx as nx

from models import Entity, FraudRing, Relationship

logger = logging.getLogger(__name__)


class FraudGraph:
    """In-memory fraud graph backed by NetworkX."""

    def __init__(self):
        self.graph: nx.Graph = nx.Graph()
        self.fraud_rings: list[FraudRing] = []

    def build(self, entities: list[Entity], relationships: list[Relationship]) -> None:
        """Construct graph from entities and relationships.

        Assigns unique IDs based on entity.id. Logs a warning and skips
        duplicates. Handles empty input by building an empty graph.
        """
        self.graph = nx.Graph()

        for entity in entities:
            if self.graph.has_node(entity.id):
                logger.warning(
                    "Duplicate entity ID '%s' encountered; skipping.", entity.id
                )
                continue

            self.graph.add_node(
                entity.id,
                entity_type=entity.entity_type,
                name=entity.name,
                properties=dict(entity.properties),
                created_at=datetime.now(timezone.utc).isoformat(),
            )

        for rel in relationships:
            self.graph.add_edge(
                rel.source_id,
                rel.target_id,
                rel_type=rel.rel_type,
                properties=dict(rel.properties),
            )

    def get_node(self, node_id: str) -> dict:
        """Return node properties including entity_type, name, and risk_score.

        Raises KeyError if node_id is not in the graph.
        """
        if not self.graph.has_node(node_id):
            raise KeyError(f"Node '{node_id}' not found in graph")

        data = dict(self.graph.nodes[node_id])
        # Ensure risk_score key exists (default 0.0 if not yet scored)
        if "risk_score" not in data:
            data["risk_score"] = 0.0
        return data

    def get_neighbors(self, node_id: str) -> list[str]:
        """Return IDs of directly connected nodes.

        Raises KeyError if node_id is not in the graph.
        """
        if not self.graph.has_node(node_id):
            raise KeyError(f"Node '{node_id}' not found in graph")

        return list(self.graph.neighbors(node_id))

    def export_json(self) -> dict:
        """Serialize full graph state to JSON-compatible dict.

        Returns a dict with entities, relationships, and fraud_rings.
        """
        entities = []
        for node_id, data in self.graph.nodes(data=True):
            entities.append({
                "id": node_id,
                "entity_type": data.get("entity_type", ""),
                "name": data.get("name", ""),
                "properties": data.get("properties", {}),
                "risk_score": data.get("risk_score", 0.0),
            })

        relationships = []
        for source, target, data in self.graph.edges(data=True):
            relationships.append({
                "source_id": source,
                "target_id": target,
                "rel_type": data.get("rel_type", ""),
                "properties": data.get("properties", {}),
            })

        fraud_rings = []
        for ring in self.fraud_rings:
            fraud_rings.append({
                "ring_id": ring.ring_id,
                "member_ids": list(ring.member_ids),
                "avg_risk_score": ring.avg_risk_score,
            })

        return {
            "entities": entities,
            "relationships": relationships,
            "fraud_rings": fraud_rings,
        }

    def import_json(self, data: dict) -> None:
        """Reconstruct graph from previously exported JSON dict.

        Raises ImportError with description for malformed JSON.
        """
        if not isinstance(data, dict):
            raise ImportError("Expected a dict, got: {}".format(type(data).__name__))

        if "entities" not in data:
            raise ImportError("Missing required key 'entities' in import data")

        if not isinstance(data["entities"], list):
            raise ImportError("'entities' must be a list")

        if "relationships" not in data:
            raise ImportError("Missing required key 'relationships' in import data")

        if not isinstance(data["relationships"], list):
            raise ImportError("'relationships' must be a list")

        # Rebuild the graph
        self.graph = nx.Graph()

        for entity_data in data["entities"]:
            if not isinstance(entity_data, dict):
                raise ImportError("Each entity must be a dict")
            if "id" not in entity_data:
                raise ImportError("Entity missing required field 'id'")

            node_id = entity_data["id"]
            self.graph.add_node(
                node_id,
                entity_type=entity_data.get("entity_type", ""),
                name=entity_data.get("name", ""),
                properties=entity_data.get("properties", {}),
                risk_score=entity_data.get("risk_score", 0.0),
            )

        for rel_data in data["relationships"]:
            if not isinstance(rel_data, dict):
                raise ImportError("Each relationship must be a dict")
            if "source_id" not in rel_data or "target_id" not in rel_data:
                raise ImportError(
                    "Relationship missing required field 'source_id' or 'target_id'"
                )

            self.graph.add_edge(
                rel_data["source_id"],
                rel_data["target_id"],
                rel_type=rel_data.get("rel_type", ""),
                properties=rel_data.get("properties", {}),
            )

        # Rebuild fraud rings
        self.fraud_rings = []
        for ring_data in data.get("fraud_rings", []):
            if not isinstance(ring_data, dict):
                raise ImportError("Each fraud_ring must be a dict")
            if "ring_id" not in ring_data:
                raise ImportError("Fraud ring missing required field 'ring_id'")

            self.fraud_rings.append(
                FraudRing(
                    ring_id=ring_data["ring_id"],
                    member_ids=ring_data.get("member_ids", []),
                    avg_risk_score=ring_data.get("avg_risk_score", 0.0),
                )
            )
