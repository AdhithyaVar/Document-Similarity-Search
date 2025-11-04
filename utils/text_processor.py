"""
Text Processor Module
Handles text preprocessing, cleaning, and chunking
"""

import re
import nltk
from typing import List, Optional
import logging
from collections import Counter

from config.settings import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CHUNK_LENGTH,
    MAX_CHUNK_LENGTH,
    STOPWORDS_ENABLED,
    LEMMATIZATION_ENABLED,
)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextProcessor:
    """
    Handles all text preprocessing and chunking operations
    """

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """
        Initialize TextProcessor
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize NLTK components
        self.stop_words = set(stopwords.words('english')) if STOPWORDS_ENABLED else set()
        self.lemmatizer = WordNetLemmatizer() if LEMMATIZATION_ENABLED else None

    def preprocess(self, text: str, deep_clean: bool = True) -> str:
        """
        Preprocess text for analysis
        
        Args:
            text: Input text
            deep_clean: Whether to perform deep cleaning
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        try:
            # Basic cleaning
            text = self._basic_clean(text)
            
            if deep_clean:
                # Advanced cleaning
                text = self._remove_urls(text)
                text = self._remove_emails(text)
                text = self._remove_special_chars(text)
                text = self._normalize_whitespace(text)
            
            return text
            
        except Exception as e:
            logger.error(f"Preprocessing error: {str(e)}")
            return text

    def _basic_clean(self, text: str) -> str:
        """Basic text cleaning"""
        # Convert to lowercase (optional - preserve case for better semantic understanding)
        # text = text.lower()
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Strip whitespace
        text = text.strip()
        
        return text

    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, '', text)

    def _remove_emails(self, text: str) -> str:
        """Remove email addresses"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.sub(email_pattern, '', text)

    def _remove_special_chars(self, text: str) -> str:
        """Remove special characters while preserving sentence structure"""
        # Keep alphanumeric, spaces, and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?;:\-\'\"]', '', text)
        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace"""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize line breaks
        text = re.sub(r'\n+', '\n', text)
        
        return text.strip()

    def chunk_text(self, text: str, method: str = "sentences") -> List[str]:
        """
        Split text into chunks for processing
        
        Args:
            text: Input text
            method: Chunking method ('sentences', 'tokens', 'paragraphs')
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        try:
            if method == "sentences":
                return self._chunk_by_sentences(text)
            elif method == "tokens":
                return self._chunk_by_tokens(text)
            elif method == "paragraphs":
                return self._chunk_by_paragraphs(text)
            else:
                return self._chunk_by_sentences(text)
                
        except Exception as e:
            logger.error(f"Chunking error: {str(e)}")
            # Fallback to simple splitting
            return self._simple_chunk(text)

    def _chunk_by_sentences(self, text: str) -> List[str]:
        """
        Chunk text by sentences with overlap
        
        Args:
            text: Input text
            
        Returns:
            List of sentence-based chunks
        """
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            # Check if adding sentence exceeds chunk size
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(" ".join(current_chunk))
                
                # Start new chunk with overlap
                overlap_size = int(len(current_chunk) * (self.chunk_overlap / self.chunk_size))
                current_chunk = current_chunk[-overlap_size:] if overlap_size > 0 else []
                current_length = sum(len(s.split()) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        # Filter out too small chunks
        chunks = [c for c in chunks if len(c) >= MIN_CHUNK_LENGTH]
        
        return chunks

    def _chunk_by_tokens(self, text: str) -> List[str]:
        """
        Chunk text by token count
        
        Args:
            text: Input text
            
        Returns:
            List of token-based chunks
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk)
            
            if len(chunk_text) >= MIN_CHUNK_LENGTH:
                chunks.append(chunk_text)
        
        return chunks

    def _chunk_by_paragraphs(self, text: str) -> List[str]:
        """
        Chunk text by paragraphs
        
        Args:
            text: Input text
            
        Returns:
            List of paragraph chunks
        """
        # Split by double newlines
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Filter empty paragraphs
        chunks = [p.strip() for p in paragraphs if len(p.strip()) >= MIN_CHUNK_LENGTH]
        
        return chunks

    def _simple_chunk(self, text: str) -> List[str]:
        """
        Simple fallback chunking method
        
        Args:
            text: Input text
            
        Returns:
            List of chunks
        """
        chunk_size_chars = self.chunk_size * 5  # Approximate characters per token
        chunks = []
        
        for i in range(0, len(text), chunk_size_chars):
            chunk = text[i:i + chunk_size_chars]
            if len(chunk) >= MIN_CHUNK_LENGTH:
                chunks.append(chunk)
        
        return chunks

    def extract_keywords(self, text: str, top_n: int = 20) -> List[tuple]:
        """
        Extract top keywords from text
        
        Args:
            text: Input text
            top_n: Number of top keywords to return
            
        Returns:
            List of (word, frequency) tuples
        """
        try:
            # Tokenize
            words = word_tokenize(text.lower())
            
            # Remove stopwords and non-alphabetic
            words = [w for w in words if w.isalpha() and w not in self.stop_words]
            
            # Lemmatize if enabled
            if self.lemmatizer:
                words = [self.lemmatizer.lemmatize(w) for w in words]
            
            # Count frequencies
            word_freq = Counter(words)
            
            return word_freq.most_common(top_n)
            
        except Exception as e:
            logger.error(f"Keyword extraction error: {str(e)}")
            return []

    def get_statistics(self, text: str) -> dict:
        """
        Get text statistics
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with text statistics
        """
        try:
            # Tokenization
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
            
            # Count unique words
            unique_words = set(w.lower() for w in words if w.isalpha())
            
            # Calculate statistics
            stats = {
                "char_count": len(text),
                "word_count": len(words),
                "sentence_count": len(sentences),
                "unique_words": len(unique_words),
                "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
                "avg_sentence_length": len(words) / len(sentences) if sentences else 0,
                "lexical_diversity": len(unique_words) / len(words) if words else 0,
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Statistics calculation error: {str(e)}")
            return {}

    def find_similar_sentences(self, text1: str, text2: str, threshold: float = 0.7) -> List[tuple]:
        """
        Find similar sentences between two texts (simple string matching)
        
        Args:
            text1: First text
            text2: Second text
            threshold: Similarity threshold
            
        Returns:
            List of (sentence1, sentence2, similarity) tuples
        """
        try:
            sentences1 = sent_tokenize(text1)
            sentences2 = sent_tokenize(text2)
            
            similar_pairs = []
            
            for s1 in sentences1:
                for s2 in sentences2:
                    # Simple word overlap similarity
                    words1 = set(word_tokenize(s1.lower()))
                    words2 = set(word_tokenize(s2.lower()))
                    
                    if words1 and words2:
                        similarity = len(words1 & words2) / len(words1 | words2)
                        
                        if similarity >= threshold:
                            similar_pairs.append((s1, s2, similarity))
            
            # Sort by similarity
            similar_pairs.sort(key=lambda x: x[2], reverse=True)
            
            return similar_pairs
            
        except Exception as e:
            logger.error(f"Similar sentence finding error: {str(e)}")
            return []

    def clean_for_embedding(self, text: str) -> str:
        """
        Clean text specifically for embedding generation
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text optimized for embeddings
        """
        # Keep most information but remove noise
        text = self._remove_urls(text)
        text = self._remove_emails(text)
        text = self._normalize_whitespace(text)
        
        return text