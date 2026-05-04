"""Data Loader for the Fraud Graph Demo.

Handles parsing of CSV and JSON input files into entities and relationships,
and provides synthetic data generation with configurable seed and size.
"""

import csv
import json
import os
import random
from typing import Optional

from models import DataLoadError, Entity, Relationship

VALID_ENTITY_TYPES = {
    "Person",
    "Account",
    "Transaction",
    "Address",
    "Phone_Number",
    "IP_Address",
    "Device",
}

VALID_RELATIONSHIP_TYPES = {
    "owns",
    "sent_to",
    "received_from",
    "used",
    "lives_at",
    "contacted",
}


def load_from_file(filepath: str) -> tuple[list[Entity], list[Relationship]]:
    """Parse CSV or JSON file into entities and relationships.

    Raises DataLoadError with descriptive message on malformed input.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".csv":
        return _load_csv(filepath)
    elif ext == ".json":
        return _load_json(filepath)
    else:
        raise DataLoadError(
            f"Unsupported file format: {ext}. Use .csv or .json"
        )


def _load_csv(filepath: str) -> tuple[list[Entity], list[Relationship]]:
    """Parse a CSV file with a record_type column distinguishing entities from relationships."""
    entities = []
    relationships = []

    with open(filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise DataLoadError("CSV file is empty or has no header row")

        if "record_type" not in reader.fieldnames:
            raise DataLoadError(
                "Missing required field 'record_type' in CSV header"
            )

        for row_num, row in enumerate(reader, start=2):  # row 1 is header
            record_type = row.get("record_type", "").strip().lower()

            if record_type == "entity":
                entity = _parse_entity_row(row, row_num)
                entities.append(entity)
            elif record_type == "relationship":
                relationship = _parse_relationship_row(row, row_num)
                relationships.append(relationship)
            else:
                raise DataLoadError(
                    f"Unknown record_type '{row.get('record_type', '')}' in row {row_num}. "
                    f"Expected 'entity' or 'relationship'"
                )

    # Validate relationship references
    entity_ids = {e.id for e in entities}
    for rel in relationships:
        if rel.source_id not in entity_ids:
            raise DataLoadError(
                f"Relationship references unknown entity '{rel.source_id}'"
            )
        if rel.target_id not in entity_ids:
            raise DataLoadError(
                f"Relationship references unknown entity '{rel.target_id}'"
            )

    return entities, relationships


def _parse_entity_row(row: dict, row_num: int) -> Entity:
    """Parse a single entity row from CSV."""
    # Check required fields
    if not row.get("id"):
        raise DataLoadError(f"Missing required field 'id' in row {row_num}")
    if not row.get("entity_type"):
        raise DataLoadError(
            f"Missing required field 'entity_type' in row {row_num}"
        )
    if not row.get("name"):
        raise DataLoadError(f"Missing required field 'name' in row {row_num}")

    entity_type = row["entity_type"].strip()
    if entity_type not in VALID_ENTITY_TYPES:
        raise DataLoadError(
            f"Unknown entity type '{entity_type}'. "
            f"Valid types: {', '.join(sorted(VALID_ENTITY_TYPES))}"
        )

    # Collect extra columns as properties
    known_fields = {"record_type", "id", "entity_type", "name"}
    properties = {
        k: v for k, v in row.items() if k not in known_fields and v
    }

    return Entity(
        id=row["id"].strip(),
        entity_type=entity_type,
        name=row["name"].strip(),
        properties=properties,
    )


def _parse_relationship_row(row: dict, row_num: int) -> Relationship:
    """Parse a single relationship row from CSV."""
    # Check required fields
    if not row.get("source_id"):
        raise DataLoadError(
            f"Missing required field 'source_id' in row {row_num}"
        )
    if not row.get("target_id"):
        raise DataLoadError(
            f"Missing required field 'target_id' in row {row_num}"
        )
    if not row.get("rel_type"):
        raise DataLoadError(
            f"Missing required field 'rel_type' in row {row_num}"
        )

    rel_type = row["rel_type"].strip()
    if rel_type not in VALID_RELATIONSHIP_TYPES:
        raise DataLoadError(
            f"Unknown relationship type '{rel_type}'. "
            f"Valid types: {', '.join(sorted(VALID_RELATIONSHIP_TYPES))}"
        )

    # Collect extra columns as properties
    known_fields = {"record_type", "source_id", "target_id", "rel_type"}
    properties = {
        k: v for k, v in row.items() if k not in known_fields and v
    }

    return Relationship(
        source_id=row["source_id"].strip(),
        target_id=row["target_id"].strip(),
        rel_type=rel_type,
        properties=properties,
    )


def _load_json(filepath: str) -> tuple[list[Entity], list[Relationship]]:
    """Parse a JSON file with entities and relationships arrays."""
    with open(filepath, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise DataLoadError(f"Invalid JSON format: {e}")

    if not isinstance(data, dict):
        raise DataLoadError(
            "JSON file must contain an object with 'entities' and 'relationships' arrays"
        )

    if "entities" not in data:
        raise DataLoadError("Missing required field 'entities' in JSON file")
    if "relationships" not in data:
        raise DataLoadError(
            "Missing required field 'relationships' in JSON file"
        )

    entities = []
    for i, entity_data in enumerate(data["entities"]):
        entity = _parse_entity_json(entity_data, i + 1)
        entities.append(entity)

    relationships = []
    for i, rel_data in enumerate(data["relationships"]):
        relationship = _parse_relationship_json(rel_data, i + 1)
        relationships.append(relationship)

    # Validate relationship references
    entity_ids = {e.id for e in entities}
    for rel in relationships:
        if rel.source_id not in entity_ids:
            raise DataLoadError(
                f"Relationship references unknown entity '{rel.source_id}'"
            )
        if rel.target_id not in entity_ids:
            raise DataLoadError(
                f"Relationship references unknown entity '{rel.target_id}'"
            )

    return entities, relationships


def _parse_entity_json(entity_data: dict, index: int) -> Entity:
    """Parse a single entity from JSON data."""
    if not isinstance(entity_data, dict):
        raise DataLoadError(f"Entity at index {index} must be an object")

    if "id" not in entity_data or not entity_data["id"]:
        raise DataLoadError(
            f"Missing required field 'id' in entity at index {index}"
        )
    if "entity_type" not in entity_data or not entity_data["entity_type"]:
        raise DataLoadError(
            f"Missing required field 'entity_type' in entity at index {index}"
        )
    if "name" not in entity_data or not entity_data["name"]:
        raise DataLoadError(
            f"Missing required field 'name' in entity at index {index}"
        )

    entity_type = entity_data["entity_type"].strip()
    if entity_type not in VALID_ENTITY_TYPES:
        raise DataLoadError(
            f"Unknown entity type '{entity_type}'. "
            f"Valid types: {', '.join(sorted(VALID_ENTITY_TYPES))}"
        )

    properties = entity_data.get("properties", {})
    if not isinstance(properties, dict):
        properties = {}

    return Entity(
        id=str(entity_data["id"]).strip(),
        entity_type=entity_type,
        name=entity_data["name"].strip(),
        properties=properties,
    )


def _parse_relationship_json(rel_data: dict, index: int) -> Relationship:
    """Parse a single relationship from JSON data."""
    if not isinstance(rel_data, dict):
        raise DataLoadError(
            f"Relationship at index {index} must be an object"
        )

    if "source_id" not in rel_data or not rel_data["source_id"]:
        raise DataLoadError(
            f"Missing required field 'source_id' in relationship at index {index}"
        )
    if "target_id" not in rel_data or not rel_data["target_id"]:
        raise DataLoadError(
            f"Missing required field 'target_id' in relationship at index {index}"
        )
    if "rel_type" not in rel_data or not rel_data["rel_type"]:
        raise DataLoadError(
            f"Missing required field 'rel_type' in relationship at index {index}"
        )

    rel_type = rel_data["rel_type"].strip()
    if rel_type not in VALID_RELATIONSHIP_TYPES:
        raise DataLoadError(
            f"Unknown relationship type '{rel_type}'. "
            f"Valid types: {', '.join(sorted(VALID_RELATIONSHIP_TYPES))}"
        )

    properties = rel_data.get("properties", {})
    if not isinstance(properties, dict):
        properties = {}

    return Relationship(
        source_id=str(rel_data["source_id"]).strip(),
        target_id=str(rel_data["target_id"]).strip(),
        rel_type=rel_type,
        properties=properties,
    )


def generate_synthetic(
    seed: Optional[int] = None, num_entities: int = 200
) -> tuple[list[Entity], list[Relationship]]:
    """Generate synthetic fraud data with at least 2 fraud ring patterns.

    Same seed produces identical output.

    Args:
        seed: Optional random seed for deterministic output.
        num_entities: Total number of entities to generate (default 200).

    Returns:
        Tuple of (entities, relationships).
    """
    rng = random.Random(seed)

    entities: list[Entity] = []
    relationships: list[Relationship] = []
    entity_id_counter = 0

    def next_id() -> str:
        nonlocal entity_id_counter
        entity_id_counter += 1
        return f"ent_{entity_id_counter:04d}"

    # --- Name pools for generating realistic-looking data ---
    first_names = [
        "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace",
        "Henry", "Iris", "Jack", "Karen", "Leo", "Mia", "Noah",
        "Olivia", "Paul", "Quinn", "Rachel", "Sam", "Tina",
    ]
    last_names = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
        "Miller", "Davis", "Rodriguez", "Martinez", "Anderson", "Taylor",
    ]
    streets = [
        "Main St", "Oak Ave", "Elm Rd", "Pine Ln", "Maple Dr",
        "Cedar Blvd", "Birch Way", "Walnut Ct", "Cherry Pl", "Ash Ter",
    ]
    cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]

    # --- Determine entity distribution ---
    # Reserve entities for fraud rings (at least 2 rings with 3-5 members each)
    num_ring_1_members = rng.randint(3, 5)
    num_ring_2_members = rng.randint(3, 5)
    ring_reserved = num_ring_1_members + num_ring_2_members
    # Each ring member needs: Person + Account = 2 entities per member
    # Plus shared attributes per ring: Address + IP_Address + Device = 3 per ring
    ring_entities_count = (ring_reserved * 2) + 6  # 2 persons * accounts + 3 shared per ring * 2 rings

    remaining = max(0, num_entities - ring_entities_count)

    # --- Generate Fraud Ring 1: Shared Address + Shared IP_Address ---
    ring1_person_ids = []
    ring1_shared_address_id = next_id()
    ring1_shared_ip_id = next_id()
    ring1_shared_device_id = next_id()

    # Shared Address entity
    entities.append(Entity(
        id=ring1_shared_address_id,
        entity_type="Address",
        name=f"{rng.randint(100, 999)} {rng.choice(streets)}, {rng.choice(cities)}",
        properties={"ring": "ring_1", "shared": "true"},
    ))
    # Shared IP_Address entity
    entities.append(Entity(
        id=ring1_shared_ip_id,
        entity_type="IP_Address",
        name=f"192.168.{rng.randint(1, 254)}.{rng.randint(1, 254)}",
        properties={"ring": "ring_1", "shared": "true"},
    ))
    # Shared Device entity
    entities.append(Entity(
        id=ring1_shared_device_id,
        entity_type="Device",
        name=f"Device-R1-{rng.randint(1000, 9999)}",
        properties={"ring": "ring_1", "shared": "true"},
    ))

    for i in range(num_ring_1_members):
        person_id = next_id()
        account_id = next_id()
        fname = rng.choice(first_names)
        lname = rng.choice(last_names)

        entities.append(Entity(
            id=person_id,
            entity_type="Person",
            name=f"{fname} {lname}",
            properties={"ring": "ring_1"},
        ))
        entities.append(Entity(
            id=account_id,
            entity_type="Account",
            name=f"Acct-{fname[0]}{lname[0]}-{rng.randint(1000, 9999)}",
            properties={"ring": "ring_1"},
        ))

        # Person owns Account
        relationships.append(Relationship(
            source_id=person_id, target_id=account_id, rel_type="owns",
            properties={},
        ))
        # Person lives_at shared Address
        relationships.append(Relationship(
            source_id=person_id, target_id=ring1_shared_address_id, rel_type="lives_at",
            properties={},
        ))
        # Person used shared IP_Address
        relationships.append(Relationship(
            source_id=person_id, target_id=ring1_shared_ip_id, rel_type="used",
            properties={},
        ))
        # Person used shared Device
        relationships.append(Relationship(
            source_id=person_id, target_id=ring1_shared_device_id, rel_type="used",
            properties={},
        ))

        ring1_person_ids.append(person_id)

    # --- Generate Fraud Ring 2: Shared Device + Shared Phone_Number ---
    ring2_person_ids = []
    ring2_shared_device_id = next_id()
    ring2_shared_phone_id = next_id()
    ring2_shared_ip_id = next_id()

    # Shared Device entity
    entities.append(Entity(
        id=ring2_shared_device_id,
        entity_type="Device",
        name=f"Device-R2-{rng.randint(1000, 9999)}",
        properties={"ring": "ring_2", "shared": "true"},
    ))
    # Shared Phone_Number entity
    entities.append(Entity(
        id=ring2_shared_phone_id,
        entity_type="Phone_Number",
        name=f"+1-555-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}",
        properties={"ring": "ring_2", "shared": "true"},
    ))
    # Shared IP_Address entity
    entities.append(Entity(
        id=ring2_shared_ip_id,
        entity_type="IP_Address",
        name=f"10.0.{rng.randint(1, 254)}.{rng.randint(1, 254)}",
        properties={"ring": "ring_2", "shared": "true"},
    ))

    for i in range(num_ring_2_members):
        person_id = next_id()
        account_id = next_id()
        fname = rng.choice(first_names)
        lname = rng.choice(last_names)

        entities.append(Entity(
            id=person_id,
            entity_type="Person",
            name=f"{fname} {lname}",
            properties={"ring": "ring_2"},
        ))
        entities.append(Entity(
            id=account_id,
            entity_type="Account",
            name=f"Acct-{fname[0]}{lname[0]}-{rng.randint(1000, 9999)}",
            properties={"ring": "ring_2"},
        ))

        # Person owns Account
        relationships.append(Relationship(
            source_id=person_id, target_id=account_id, rel_type="owns",
            properties={},
        ))
        # Person used shared Device
        relationships.append(Relationship(
            source_id=person_id, target_id=ring2_shared_device_id, rel_type="used",
            properties={},
        ))
        # Person contacted shared Phone_Number
        relationships.append(Relationship(
            source_id=person_id, target_id=ring2_shared_phone_id, rel_type="contacted",
            properties={},
        ))
        # Person used shared IP_Address
        relationships.append(Relationship(
            source_id=person_id, target_id=ring2_shared_ip_id, rel_type="used",
            properties={},
        ))

        ring2_person_ids.append(person_id)

    # --- Generate remaining legitimate entities ---
    legitimate_person_ids = []
    legitimate_account_ids = []
    all_address_ids = [ring1_shared_address_id]
    all_ip_ids = [ring1_shared_ip_id, ring2_shared_ip_id]
    all_device_ids = [ring1_shared_device_id, ring2_shared_device_id]
    all_phone_ids = [ring2_shared_phone_id]

    # Distribute remaining entities across types
    if remaining > 0:
        # Allocate: ~30% Person, ~20% Account, ~15% Transaction, ~10% Address,
        # ~10% Phone_Number, ~8% IP_Address, ~7% Device
        type_weights = [
            ("Person", 0.30),
            ("Account", 0.20),
            ("Transaction", 0.15),
            ("Address", 0.10),
            ("Phone_Number", 0.10),
            ("IP_Address", 0.08),
            ("Device", 0.07),
        ]

        for entity_type, weight in type_weights:
            count = max(1, int(remaining * weight))
            for _ in range(count):
                eid = next_id()
                if entity_type == "Person":
                    name = f"{rng.choice(first_names)} {rng.choice(last_names)}"
                    legitimate_person_ids.append(eid)
                elif entity_type == "Account":
                    name = f"Acct-{rng.randint(10000, 99999)}"
                    legitimate_account_ids.append(eid)
                elif entity_type == "Transaction":
                    name = f"TX-{rng.randint(100000, 999999)}"
                elif entity_type == "Address":
                    name = f"{rng.randint(1, 9999)} {rng.choice(streets)}, {rng.choice(cities)}"
                    all_address_ids.append(eid)
                elif entity_type == "Phone_Number":
                    name = f"+1-{rng.randint(200, 999)}-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}"
                    all_phone_ids.append(eid)
                elif entity_type == "IP_Address":
                    name = f"{rng.randint(1, 223)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
                    all_ip_ids.append(eid)
                elif entity_type == "Device":
                    name = f"Device-{rng.randint(10000, 99999)}"
                    all_device_ids.append(eid)
                else:
                    name = f"{entity_type}-{eid}"

                entities.append(Entity(
                    id=eid,
                    entity_type=entity_type,
                    name=name,
                    properties={},
                ))

    # --- Generate relationships for legitimate entities ---
    # Person owns Account
    for person_id in legitimate_person_ids:
        if legitimate_account_ids:
            acct_id = rng.choice(legitimate_account_ids)
            relationships.append(Relationship(
                source_id=person_id, target_id=acct_id, rel_type="owns",
                properties={},
            ))

    # Person lives_at Address
    for person_id in legitimate_person_ids:
        if all_address_ids:
            addr_id = rng.choice(all_address_ids)
            relationships.append(Relationship(
                source_id=person_id, target_id=addr_id, rel_type="lives_at",
                properties={},
            ))

    # Some Persons use IP/Device
    for person_id in legitimate_person_ids[:len(legitimate_person_ids) // 2]:
        if all_ip_ids:
            ip_id = rng.choice(all_ip_ids)
            relationships.append(Relationship(
                source_id=person_id, target_id=ip_id, rel_type="used",
                properties={},
            ))
        if all_device_ids:
            dev_id = rng.choice(all_device_ids)
            relationships.append(Relationship(
                source_id=person_id, target_id=dev_id, rel_type="used",
                properties={},
            ))

    # Some Account sent_to/received_from transactions
    all_account_ids = legitimate_account_ids[:]
    transaction_entities = [e for e in entities if e.entity_type == "Transaction"]
    for tx in transaction_entities:
        if all_account_ids and len(all_account_ids) >= 2:
            src_acct = rng.choice(all_account_ids)
            dst_acct = rng.choice([a for a in all_account_ids if a != src_acct] or all_account_ids)
            relationships.append(Relationship(
                source_id=src_acct, target_id=tx.id, rel_type="sent_to",
                properties={"amount": str(rng.randint(10, 10000))},
            ))
            relationships.append(Relationship(
                source_id=tx.id, target_id=dst_acct, rel_type="received_from",
                properties={"amount": str(rng.randint(10, 10000))},
            ))

    # Some Persons contacted Phone_Numbers
    for person_id in legitimate_person_ids[len(legitimate_person_ids) // 2:]:
        if all_phone_ids:
            phone_id = rng.choice(all_phone_ids)
            relationships.append(Relationship(
                source_id=person_id, target_id=phone_id, rel_type="contacted",
                properties={},
            ))

    return entities, relationships
