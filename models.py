"""Data models for the Fraud Graph Demo."""

from dataclasses import dataclass, field


@dataclass
class Entity:
    """A node in the fraud graph representing a person, account, transaction, etc."""

    id: str  # Unique identifier (UUID or user-provided)
    entity_type: str  # Person | Account | Transaction | Address | Phone_Number | IP_Address | Device
    name: str  # Display name
    properties: dict = field(default_factory=dict)  # Additional key-value properties


@dataclass
class Relationship:
    """An edge in the fraud graph connecting two entities."""

    source_id: str  # Entity ID of source node
    target_id: str  # Entity ID of target node
    rel_type: str  # owns | sent_to | received_from | used | lives_at | contacted
    properties: dict = field(default_factory=dict)  # Optional properties (amount, timestamp, etc.)


@dataclass
class FraudRing:
    """A cluster of entities connected through shared attributes that exhibit suspicious patterns."""

    ring_id: str  # Unique ring identifier
    member_ids: list[str] = field(default_factory=list)  # Entity IDs in this ring
    avg_risk_score: float = 0.0  # Average risk score of members


class DataLoadError(Exception):
    """Raised when data loading fails due to malformed input, missing fields, or invalid structure."""

    pass
