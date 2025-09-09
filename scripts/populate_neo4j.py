import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_processing.document_loader import DocumentLoader
from src.graph_db.neo4j_manager import Neo4jManager


CONCEPTS = [
    {"name": "Arduino", "description": "Popular microcontroller board"},
    {"name": "Raspberry Pi", "description": "Single board computer"},
    {"name": "LED", "description": "Light-emitting diode"},
    {"name": "Ohm's Law", "description": "Voltage-current-resistance relationship"},
    {"name": "Sensor", "description": "Device detecting environment changes"},
    {"name": "Microcontroller", "description": "Integrated digital computing device"},
    {"name": "Actuator", "description": "Device that moves or controls"},
]

if __name__ == "__main__":
    loader = DocumentLoader()
    chunks = loader.load_documents()

    all_concepts = {c['name']: c for c in CONCEPTS}
    for chunk in chunks:
        if 'concepts' in chunk:
            for cname in chunk['concepts']:
                if cname not in all_concepts:
                    all_concepts[cname] = {"name": cname, "description": ""}

    manager = Neo4jManager()
    manager.create_concept_nodes(list(all_concepts.values()))
    manager.create_chunk_nodes(chunks)
    for chunk in chunks:
        if 'concepts' in chunk:
            manager.link_chunk_to_concept(chunk, chunk['concepts'])
    manager.close()
    print("Neo4j graph populated!")
