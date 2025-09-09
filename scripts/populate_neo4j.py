# scripts/populate_neo4j.py
from src.data_processing.document_loader import DocumentLoader
from src.graph_db.neo4j_manager import Neo4jManager

# Add your master concept list here
CONCEPTS = [
    {"name": "Arduino", "description": "Popular microcontroller board"},
    {"name": "Raspberry Pi", "description": "Single board computer for projects"},
    {"name": "LED", "description": "Light-emitting diode"},
    {"name": "Ohm's Law", "description": "Relationship between voltage, current, resistance"},
    {"name": "Sensor", "description": "Detects events/changes in environment"},
    {"name": "Microcontroller", "description": "Integrated computing device"},
    {"name": "Actuator", "description": "Device that moves or controls systems"},
]

if __name__ == "__main__":
    # Load processed chunks
    loader = DocumentLoader()
    chunks = loader.load_documents()
    # Extract unique concepts mentioned
    all_concepts = {c['name']: c for c in CONCEPTS}
    for chunk in chunks:
        if 'concepts' in chunk:
            for cname in chunk['concepts']:
                all_concepts.setdefault(cname, {"name": cname, "description": ""})
    manager = Neo4jManager()
    manager.create_concept_nodes(list(all_concepts.values()))
    manager.create_chunk_nodes(chunks)
    for chunk in chunks:
        if 'concepts' in chunk:
            manager.link_chunk_to_concept(chunk, chunk['concepts'])
    manager.close()
    print("✅ Knowledge graph populated with concepts and chunks.")
