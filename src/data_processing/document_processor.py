import re
from typing import List, Dict

class DocumentProcessor:
    def __init__(self, chunk_size: int = 400, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def clean_text(self, text: str) -> str:
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        lines = [line.strip() for line in text.split('\n')]
        return '\n'.join(lines).strip()

    def smart_chunk_text(self, text: str) -> List[Dict]:
        clean_text = self.clean_text(text)
        paragraphs = clean_text.split('\n\n')
        chunks = []
        current_chunk = ""
        current_word_count = 0
        chunk_id = 0
        for para in paragraphs:
            words = para.split()
            word_len = len(words)
            if current_word_count + word_len > self.chunk_size and current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'chunk_id': chunk_id,
                    'word_count': current_word_count,
                    'section_title': 'General'
                })
                chunk_id += 1
                current_chunk = para + " "
                current_word_count = word_len
            else:
                current_chunk += para + " "
                current_word_count += word_len
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'chunk_id': chunk_id,
                'word_count': current_word_count,
                'section_title': 'General'
            })
        return chunks
