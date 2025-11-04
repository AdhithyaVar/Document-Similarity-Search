"""
Similarity Engine Module
Core semantic similarity and plagiarism detection logic
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from typing import List, Dict, Tuple
import logging
import plotly.graph_objects as go
import plotly.express as px

from config.settings import (
    MODEL_NAME,
    SIMILARITY_THRESHOLDS,
    BATCH_SIZE,
    CACHE_EMBEDDINGS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimilarityEngine:
    """
    Handles semantic similarity computation and analysis
    """

    def __init__(self):
        """Initialize SimilarityEngine with sentence transformer model"""
        self.model = self._load_model()
        self.thresholds = SIMILARITY_THRESHOLDS

    @st.cache_resource
    def _load_model(_self):
        """
        Load sentence transformer model (cached)
        
        Returns:
            Loaded model
        """
        try:
            logger.info(f"Loading model: {MODEL_NAME}")
            model = SentenceTransformer(MODEL_NAME)
            logger.info("Model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Model loading error: {str(e)}")
            raise

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a single text
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        try:
            embedding = self.model.encode([text], show_progress_bar=False)[0]
            return embedding
        except Exception as e:
            logger.error(f"Embedding generation error: {str(e)}")
            return np.zeros(384)  # Return zero vector on error

    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for multiple texts
        
        Args:
            texts: List of texts
            
        Returns:
            Array of embeddings
        """
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=BATCH_SIZE,
                show_progress_bar=True if len(texts) > 10 else False,
            )
            return embeddings
        except Exception as e:
            logger.error(f"Batch embedding error: {str(e)}")
            return np.zeros((len(texts), 384))

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            emb1: First embedding
            emb2: Second embedding
            
        Returns:
            Similarity score (0-1)
        """
        try:
            similarity = cosine_similarity([emb1], [emb2])[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"Similarity computation error: {str(e)}")
            return 0.0

    def analyze(
        self,
        text1: str,
        text2: str,
        chunks1: List[str],
        chunks2: List[str],
        threshold: float = 0.7,
        analysis_mode: str = "Standard",
    ) -> Dict:
        """
        Perform comprehensive similarity analysis
        
        Args:
            text1: First document text
            text2: Second document text
            chunks1: Chunks from first document
            chunks2: Chunks from second document
            threshold: Similarity threshold
            analysis_mode: Analysis depth
            
        Returns:
            Analysis results dictionary
        """
        try:
            logger.info(f"Starting {analysis_mode} analysis")
            
            # Overall document similarity
            emb1_full = self.get_embedding(text1)
            emb2_full = self.get_embedding(text2)
            overall_similarity = self.compute_similarity(emb1_full, emb2_full)
            
            # Chunk-level analysis
            chunk_results = self._analyze_chunks(chunks1, chunks2, threshold)
            
            # Generate heatmap
            heatmap_data = self._generate_heatmap(chunks1, chunks2)
            
            # Find matching segments
            matching_segments = self._find_matching_segments(
                chunks1, chunks2, chunk_results, top_n=10
            )
            
            # Determine risk level
            risk_level = self._determine_risk_level(overall_similarity)
            
            # Calculate statistics
            doc1_stats = self._calculate_doc_stats(text1)
            doc2_stats = self._calculate_doc_stats(text2)
            
            # Generate interpretation
            interpretation = self._generate_interpretation(
                overall_similarity, len(matching_segments), risk_level
            )
            
            # Compile results
            results = {
                "overall_similarity": overall_similarity,
                "plagiarism_score": overall_similarity * 100,
                "risk_level": risk_level,
                "matching_segments": len(matching_segments),
                "matching_segments_detail": matching_segments,
                "unique_percentage": 1 - overall_similarity,
                "chunk_similarities": chunk_results,
                "heatmap_data": heatmap_data,
                "doc1_stats": doc1_stats,
                "doc2_stats": doc2_stats,
                "interpretation": interpretation,
                "threshold_used": threshold,
                "analysis_mode": analysis_mode,
            }
            
            logger.info(f"Analysis complete. Similarity: {overall_similarity:.2%}")
            
            return results
            
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            raise

    def _analyze_chunks(
        self, chunks1: List[str], chunks2: List[str], threshold: float
    ) -> List[Dict]:
        """
        Analyze similarity between chunk pairs
        
        Args:
            chunks1: Chunks from document 1
            chunks2: Chunks from document 2
            threshold: Similarity threshold
            
        Returns:
            List of chunk comparison results
        """
        try:
            # Get embeddings for all chunks
            embeddings1 = self.get_embeddings_batch(chunks1)
            embeddings2 = self.get_embeddings_batch(chunks2)
            
            # Compute pairwise similarities
            similarity_matrix = cosine_similarity(embeddings1, embeddings2)
            
            results = []
            for i, emb1 in enumerate(embeddings1):
                for j, emb2 in enumerate(embeddings2):
                    similarity = similarity_matrix[i][j]
                    
                    if similarity >= threshold:
                        results.append({
                            "chunk1_idx": i,
                            "chunk2_idx": j,
                            "chunk1_text": chunks1[i][:200],  # Preview
                            "chunk2_text": chunks2[j][:200],
                            "similarity": float(similarity),
                        })
            
            # Sort by similarity
            results.sort(key=lambda x: x["similarity"], reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Chunk analysis error: {str(e)}")
            return []

    def _generate_heatmap(self, chunks1: List[str], chunks2: List[str]) -> go.Figure:
        """
        Generate similarity heatmap
        
        Args:
            chunks1: Chunks from document 1
            chunks2: Chunks from document 2
            
        Returns:
            Plotly figure
        """
        try:
            # Limit chunks for visualization (max 50x50)
            max_chunks = 50
            chunks1_sample = chunks1[:max_chunks]
            chunks2_sample = chunks2[:max_chunks]
            
            # Get embeddings
            embeddings1 = self.get_embeddings_batch(chunks1_sample)
            embeddings2 = self.get_embeddings_batch(chunks2_sample)
            
            # Compute similarity matrix
            similarity_matrix = cosine_similarity(embeddings1, embeddings2)
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=similarity_matrix,
                x=[f"C2-{i+1}" for i in range(len(chunks2_sample))],
                y=[f"C1-{i+1}" for i in range(len(chunks1_sample))],
                colorscale="RdYlGn",
                colorbar=dict(title="Similarity"),
                hovertemplate="Doc1 Chunk: %{y}<br>Doc2 Chunk: %{x}<br>Similarity: %{z:.2f}<extra></extra>",
            ))
            
            fig.update_layout(
                title="Chunk-to-Chunk Similarity Heatmap",
                xaxis_title="Document 2 Chunks",
                yaxis_title="Document 1 Chunks",
                height=600,
                width=800,
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Heatmap generation error: {str(e)}")
            return go.Figure()

    def _find_matching_segments(
        self,
        chunks1: List[str],
        chunks2: List[str],
        chunk_results: List[Dict],
        top_n: int = 10,
    ) -> List[Dict]:
        """
        Find and format top matching segments
        
        Args:
            chunks1: Chunks from document 1
            chunks2: Chunks from document 2
            chunk_results: Chunk comparison results
            top_n: Number of top matches to return
            
        Returns:
            List of matching segment details
        """
        try:
            matching_segments = []
            
            for result in chunk_results[:top_n]:
                idx1 = result["chunk1_idx"]
                idx2 = result["chunk2_idx"]
                
                segment = {
                    "text1": chunks1[idx1][:500],  # Limit length
                    "text2": chunks2[idx2][:500],
                    "score": result["similarity"],
                    "position1": idx1,
                    "position2": idx2,
                }
                
                matching_segments.append(segment)
            
            return matching_segments
            
        except Exception as e:
            logger.error(f"Matching segments error: {str(e)}")
            return []

    def _determine_risk_level(self, similarity: float) -> str:
        """
        Determine plagiarism risk level
        
        Args:
            similarity: Overall similarity score
            
        Returns:
            Risk level string
        """
        if similarity >= self.thresholds["high"]:
            return "High"
        elif similarity >= self.thresholds["moderate"]:
            return "Moderate"
        else:
            return "Low"

    def _calculate_doc_stats(self, text: str) -> Dict:
        """
        Calculate document statistics
        
        Args:
            text: Document text
            
        Returns:
            Statistics dictionary
        """
        try:
            words = text.split()
            sentences = text.count('.') + text.count('!') + text.count('?')
            
            return {
                "char_count": len(text),
                "word_count": len(words),
                "sentence_count": max(sentences, 1),
                "unique_words": len(set(w.lower() for w in words)),
                "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
            }
        except Exception as e:
            logger.error(f"Statistics calculation error: {str(e)}")
            return {}

    def _generate_interpretation(
        self, similarity: float, matching_count: int, risk_level: str
    ) -> str:
        """
        Generate human-readable interpretation
        
        Args:
            similarity: Overall similarity score
            matching_count: Number of matching segments
            risk_level: Risk level
            
        Returns:
            Interpretation string
        """
        interpretations = {
            "High": f"The documents show {similarity:.1%} similarity, indicating substantial content overlap. "
                   f"Found {matching_count} highly similar segments. This suggests significant plagiarism risk. "
                   "Manual review recommended to verify originality and proper attribution.",
            
            "Moderate": f"The documents show {similarity:.1%} similarity with {matching_count} matching segments. "
                       "This level of similarity could indicate shared topic or common research area. "
                       "Review matching sections to determine if similarities are appropriately cited or constitute plagiarism.",
            
            "Low": f"The documents show only {similarity:.1%} similarity with {matching_count} minor overlaps. "
                  "This suggests the documents are largely original or discuss different aspects of a topic. "
                  "The content appears to be distinct with minimal plagiarism concern.",
        }
        
        return interpretations.get(risk_level, "Unable to generate interpretation.")

    def compare_multiple(self, documents: List[str]) -> Dict:
        """
        Compare multiple documents (future feature)
        
        Args:
            documents: List of document texts
            
        Returns:
            Comparison results
        """
        # Placeholder for batch comparison feature
        logger.info("Multi-document comparison not yet implemented")
        return {"error": "Feature coming soon"}