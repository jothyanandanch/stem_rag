# src/graph_db/neo4j_manager.py
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

class Neo4jManager:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "test"))
        )

    def close(self):
        self.driver.close()

    def create_concept_nodes(self, concepts: List[Dict]):
        with self.driver.session() as session:
            for concept in concepts:
                session.run(
                    """
                    MERGE (c:Concept {name: $name})
                    ON CREATE SET c.description = $description
                    """,
                    name=concept["name"],
                    description=concept.get("description", ""),
                )

    def create_chunk_nodes(self, chunks: List[Dict]):
        with self.driver.session() as session:
            for chunk in chunks:
                session.run(
                    """
                    MERGE (ch:Chunk {chunk_id: $chunk_id, topic: $topic})
                    SET ch.text = $text, ch.section_title = $section_title
                    """,
                    chunk_id=chunk["chunk_id"],
                    topic=chunk["topic"],
                    text=chunk["text"],
                    section_title=chunk.get("section_title", ""),
                )

    def link_chunk_to_concept(self, chunk: Dict, concepts: List[str]):
        with self.driver.session() as session:
            for concept_name in concepts:
                session.run(
                    """
                    MATCH (c:Concept {name: $concept_name}), (ch:Chunk {chunk_id: $chunk_id, topic: $topic})
                    MERGE (c)-[:MENTIONED_IN]->(ch)
                    """,
                    concept_name=concept_name,
                    chunk_id=chunk["chunk_id"],
                    topic=chunk["topic"],
                )

    def find_chunks_for_concept(self, concept_name: str) -> List[Dict]:
        with self.driver.session() as session:
            results = session.run(
                """
                MATCH (c:Concept {name: $concept_name})-[:MENTIONED_IN]->(ch:Chunk)
                RETURN ch.section_title AS section, ch.text AS text
                """,
                concept_name=concept_name,
            )
            return [{"section": r["section"], "text": r["text"]} for r in results]

if __name__ == "__main__":
    test_concepts = [{"name": "Arduino", "description": "Open-source microcontroller board."}]
    test_chunks = [{"chunk_id": 0, "topic": "Arduino", "text": "Arduino is an open-source ...", "section_title": "What is Arduino?"}]
    manager = Neo4jManager()
    manager.create_concept_nodes(test_concepts)
    manager.create_chunk_nodes(test_chunks)
    manager.link_chunk_to_concept(test_chunks, ["Arduino"])
    print("Chunks for 'Arduino':", manager.find_chunks_for_concept("Arduino"))
    manager.close()
