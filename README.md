# Fraud Graph Demo

A lightweight, zero-cost interactive fraud graph visualization that runs entirely locally. Build a graph from your own data or synthetic data, score entities for fraud risk, detect fraud rings, and explore everything in an interactive web UI.

## Quick Start

```bash
# 1. Install dependencies
pip3 install -r requirements.txt

# 2. Run with synthetic data
python3 app.py

# 3. Open in browser
# http://127.0.0.1:5000
```

## Requirements

- Python 3.9+
- No cloud services, no API keys, no Docker — everything runs locally

## Usage

### Synthetic data (default)

```bash
python3 app.py
```

Options:
- `--seed 42` — random seed for reproducible data (default: 42)
- `--entities 200` — number of entities to generate (default: 200)
- `--port 5000` — server port (default: 5000)
- `--debug` — enable Flask debug mode

```bash
# Example: smaller graph on a different port
python3 app.py --entities 50 --seed 123 --port 8080
```

### Your own data

```bash
python3 app.py --data path/to/your/data.json
```

#### JSON format

```json
{
  "entities": [
    {"id": "p1", "entity_type": "Person", "name": "Alice", "properties": {"age": "30"}},
    {"id": "a1", "entity_type": "Account", "name": "Acct-001", "properties": {}}
  ],
  "relationships": [
    {"source_id": "p1", "target_id": "a1", "rel_type": "owns", "properties": {}}
  ]
}
```

#### CSV format

```csv
record_type,id,entity_type,name,source_id,target_id,rel_type
entity,p1,Person,Alice,,,
entity,a1,Account,Acct-001,,,
relationship,,,,p1,a1,owns
```

#### Supported entity types
Person, Account, Transaction, Address, Phone_Number, IP_Address, Device

#### Supported relationship types
owns, sent_to, received_from, used, lives_at, contacted

## Features

- **Interactive graph** — force-directed layout with zoom, pan, and hover tooltips
- **Color-coded nodes** — each entity type has a distinct color
- **Risk scoring** — entities scored 0.0–1.0 based on graph topology; higher risk = larger node
- **Fraud ring detection** — clusters sharing 2+ attributes are flagged with red borders
- **Search & filter** — filter by entity type, risk tier, or search by name/ID
- **Node selection** — enter specific node IDs to isolate a subgraph, optionally including neighbors
- **Export/import** — save and reload graph state as JSON

## Running Tests

```bash
pip3 install -r requirements.txt
python3 -m pytest tests/ -v
```

## Project Structure

```
├── app.py              # Flask server + CLI entry point
├── data_loader.py      # CSV/JSON parsing + synthetic data generation
├── graph_engine.py     # NetworkX graph wrapper (build, query, export/import)
├── fraud_scorer.py     # Risk scoring + fraud ring detection
├── visualizer.py       # PyVis graph rendering + filtering
├── models.py           # Data models (Entity, Relationship, FraudRing)
├── requirements.txt    # Python dependencies
├── templates/          # Generated HTML (PyVis output + wrapper page)
│   └── index.html      # Main UI with sidebar + graph iframe
└── tests/              # Unit tests (138 tests)
```

## Tech Stack

- **NetworkX** — in-memory graph engine
- **Flask** — lightweight web server
- **PyVis** — interactive graph visualization (generates HTML from NetworkX)
- **pytest + hypothesis** — testing (unit + property-based)
