# src/data_processing/document_processor.py
import re
from typing import List, Dict, Optional

class DocumentProcessor:
    """Enhanced document processing for STEM content with better chunking"""
    
    def __init__(self, chunk_size: int = 400, overlap: int = 50, min_chunk_size: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text while preserving structure"""
        # Remove markdown headers but keep the content
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
        
        # Remove multiple consecutive newlines but keep paragraph breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove extra spaces but preserve single spaces
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Remove trailing/leading whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Remove empty lines at start and end
        text = text.strip()
        
        return text
    
    def extract_sections(self, text: str) -> List[Dict]:
        """Extract sections based on headers and structure"""
        sections = []
        
        # Split by double newlines to get paragraphs
        paragraphs = text.split('\n\n')
        
        current_section = {
            'title': 'Introduction',
            'content': '',
            'level': 1
        }
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Check if this looks like a header (starts with **text:** or has special patterns)
            if self._is_header(para):
                # Save previous section if it has content
                if current_section['content'].strip():
                    sections.append(current_section.copy())
                
                # Start new section
                current_section = {
                    'title': self._extract_title(para),
                    'content': para + '\n\n',
                    'level': self._get_header_level(para)
                }
            else:
                # Add to current section
                current_section['content'] += para + '\n\n'
        
        # Add the last section
        if current_section['content'].strip():
            sections.append(current_section)
        
        return sections
    
    def _is_header(self, text: str) -> bool:
        """Check if text looks like a header"""
        # Headers often start with **text:** or are short and descriptive
        patterns = [
            r'^\*\*[^*]+:\*\*',  # **Header:**
            r'^[A-Z][^.!?]*:$',   # TITLE:
            r'^[A-Z][A-Za-z\s]{5,30}:$',  # Descriptive Title:
        ]
        
        for pattern in patterns:
            if re.match(pattern, text.strip()):
                return True
        
        # Also check if it's a short line that doesn't end with punctuation
        if len(text.strip()) < 100 and not text.strip().endswith(('.', '!', '?', ':')):
            words = text.strip().split()
            if 2 <= len(words) <= 8:  # Reasonable header length
                return True
        
        return False
    
    def _extract_title(self, text: str) -> str:
        """Extract clean title from header text"""
        # Remove markdown formatting
        title = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        # Remove trailing colons
        title = title.rstrip(':').strip()
        # Take first line only
        title = title.split('\n')[0]
        return title
    
    def _get_header_level(self, text: str) -> int:
        """Determine header level (1-3)"""
        if text.startswith('**') and text.count('**') >= 2:
            return 2
        elif ':' in text:
            return 2
        else:
            return 1
    
    def smart_chunk_text(self, text: str, preserve_sections: bool = True) -> List[Dict]:
        """Smart chunking that respects document structure"""
        clean_text = self.clean_text(text)
        
        if preserve_sections:
            sections = self.extract_sections(clean_text)
            chunks = []
            
            for section in sections:
                section_chunks = self._chunk_section(section)
                chunks.extend(section_chunks)
            
            return chunks
        else:
            # Fallback to simple chunking
            return self.chunk_text(clean_text)
    
    def _chunk_section(self, section: Dict) -> List[Dict]:
        """Chunk a single section while preserving context"""
        content = section['content']
        title = section['title']
        
        # If section is small enough, keep as one chunk
        word_count = len(content.split())
        if word_count <= self.chunk_size:
            return [{
                'text': content,
                'chunk_id': 0,
                'word_count': word_count,
                'section_title': title,
                'section_level': section['level'],
                'is_complete_section': True
            }]
        
        # Split into sentences for better chunking
        sentences = self._split_into_sentences(content)
        chunks = []
        current_chunk = ""
        current_word_count = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            # If adding this sentence would exceed chunk size, save current chunk
            if current_word_count + sentence_words > self.chunk_size and current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'chunk_id': chunk_id,
                    'word_count': current_word_count,
                    'section_title': title,
                    'section_level': section['level'],
                    'is_complete_section': False
                })
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + sentence + " "
                current_word_count = len(current_chunk.split())
                chunk_id += 1
            else:
                current_chunk += sentence + " "
                current_word_count += sentence_words
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'chunk_id': chunk_id,
                'word_count': current_word_count,
                'section_title': title,
                'section_level': section['level'],
                'is_complete_section': False
            })
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for better chunking"""
        # Simple sentence splitting (can be improved with nltk if needed)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of current chunk"""
        words = text.split()
        if len(words) <= self.overlap:
            return text + " "
        else:
            overlap_words = words[-self.overlap:]
            return " ".join(overlap_words) + " "
    
    def chunk_text(self, text: str) -> List[Dict]:
        """Fallback simple chunking method (backward compatibility)"""
        clean_text = self.clean_text(text)
        words = clean_text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            # Skip chunks that are too small
            if len(chunk_words) < self.min_chunk_size and i > 0:
                continue
            
            chunks.append({
                'text': chunk_text,
                'chunk_id': len(chunks),
                'word_count': len(chunk_words),
                'section_title': 'General',
                'section_level': 1,
                'is_complete_section': False
            })
            
            if i + self.chunk_size >= len(words):
                break
        
        return chunks
    
    def get_chunk_preview(self, chunk: Dict, max_length: int = 100) -> str:
        """Get a preview of chunk content"""
        text = chunk['text']
        if len(text) <= max_length:
            return text
        else:
            return text[:max_length-3] + "..."