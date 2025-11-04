"""
Unit Tests for Plagiarism Detection System
Version: 2.0.0
"""

import pytest
import sys
from pathlib import Path
from io import BytesIO

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.file_handler import FileHandler
from utils.text_processor import TextProcessor
from utils.similarity_engine import SimilarityEngine
from utils.report_generator import ReportGenerator


class TestFileHandler:
    """Test cases for FileHandler"""

    def setup_method(self):
        """Setup test fixtures"""
        self.handler = FileHandler()

    def test_validate_file_size(self):
        """Test file size validation"""
        # Create a mock file
        mock_file = BytesIO(b"test content")
        mock_file.name = "test.txt"
        
        is_valid, error = self.handler.validate_file(mock_file)
        assert is_valid is True
        assert error is None

    def test_extract_text_from_txt(self):
        """Test text extraction from TXT file"""
        content = b"This is a test document with some content."
        mock_file = BytesIO(content)
        mock_file.name = "test.txt"
        
        text = self.handler.extract_text(mock_file)
        assert text is not None
        assert "test document" in text

    def test_clean_text(self):
        """Test text cleaning"""
        dirty_text = "This   has    multiple   spaces\n\n\nand newlines"
        clean = self.handler._clean_text(dirty_text)
        
        assert "  " not in clean
        assert clean.strip() == clean

    def test_invalid_file_extension(self):
        """Test handling of invalid file extensions"""
        mock_file = BytesIO(b"content")
        mock_file.name = "test.invalid"
        
        is_valid, error = self.handler.validate_file(mock_file)
        assert is_valid is False
        assert error is not None


class TestTextProcessor:
    """Test cases for TextProcessor"""

    def setup_method(self):
        """Setup test fixtures"""
        self.processor = TextProcessor(chunk_size=100, chunk_overlap=20)

    def test_preprocess_text(self):
        """Test text preprocessing"""
        text = "This is a TEST with URLs http://example.com and emails test@example.com"
        processed = self.processor.preprocess(text, deep_clean=True)
        
        assert "http://" not in processed
        assert "@" not in processed

    def test_chunk_text_by_sentences(self):
        """Test chunking by sentences"""
        text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
        chunks = self.processor.chunk_text(text, method="sentences")
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_extract_keywords(self):
        """Test keyword extraction"""
        text = "Python is a programming language. Python is used for data science. Programming is fun."
        keywords = self.processor.extract_keywords(text, top_n=5)
        
        assert len(keywords) > 0
        assert all(isinstance(kw, tuple) for kw in keywords)

    def test_get_statistics(self):
        """Test text statistics calculation"""
        text = "This is a test. It has multiple sentences."
        stats = self.processor.get_statistics(text)
        
        assert "word_count" in stats
        assert "sentence_count" in stats
        assert stats["word_count"] > 0
        assert stats["sentence_count"] > 0

    def test_empty_text_handling(self):
        """Test handling of empty text"""
        chunks = self.processor.chunk_text("")
        assert chunks == []
        
        stats = self.processor.get_statistics("")
        assert stats == {}


class TestSimilarityEngine:
    """Test cases for SimilarityEngine"""

    def setup_method(self):
        """Setup test fixtures"""
        self.engine = SimilarityEngine()

    def test_get_embedding(self):
        """Test embedding generation"""
        text = "This is a test sentence for embedding."
        embedding = self.engine.get_embedding(text)
        
        assert embedding is not None
        assert len(embedding) == 384  # all-MiniLM-L6-v2 dimension

    def test_compute_similarity_identical(self):
        """Test similarity of identical texts"""
        text = "This is a test sentence."
        emb1 = self.engine.get_embedding(text)
        emb2 = self.engine.get_embedding(text)
        
        similarity = self.engine.compute_similarity(emb1, emb2)
        assert similarity > 0.99  # Should be very close to 1.0

    def test_compute_similarity_different(self):
        """Test similarity of different texts"""
        text1 = "Python is a programming language."
        text2 = "The weather is nice today."
        
        emb1 = self.engine.get_embedding(text1)
        emb2 = self.engine.get_embedding(text2)
        
        similarity = self.engine.compute_similarity(emb1, emb2)
        assert similarity < 0.5  # Should be low

    def test_determine_risk_level(self):
        """Test risk level determination"""
        assert self.engine._determine_risk_level(0.9) == "High"
        assert self.engine._determine_risk_level(0.6) == "Moderate"
        assert self.engine._determine_risk_level(0.2) == "Low"

    def test_analyze_complete(self):
        """Test complete analysis workflow"""
        text1 = "Artificial intelligence is transforming the world. Machine learning enables computers to learn."
        text2 = "AI is changing everything. ML allows machines to improve automatically."
        
        chunks1 = ["Artificial intelligence is transforming the world.", 
                   "Machine learning enables computers to learn."]
        chunks2 = ["AI is changing everything.", 
                   "ML allows machines to improve automatically."]
        
        results = self.engine.analyze(text1, text2, chunks1, chunks2)
        
        assert "overall_similarity" in results
        assert "risk_level" in results
        assert "matching_segments" in results
        assert 0 <= results["overall_similarity"] <= 1


class TestReportGenerator:
    """Test cases for ReportGenerator"""

    def setup_method(self):
        """Setup test fixtures"""
        self.generator = ReportGenerator()
        self.mock_results = {
            "overall_similarity": 0.75,
            "plagiarism_score": 75.0,
            "risk_level": "Moderate",
            "matching_segments": 5,
            "unique_percentage": 0.25,
            "file1_name": "doc1.txt",
            "file2_name": "doc2.txt",
            "analysis_mode": "Standard",
            "threshold_used": 0.7,
            "interpretation": "Test interpretation",
            "doc1_stats": {
                "word_count": 100,
                "sentence_count": 5,
                "unique_words": 80,
            },
            "doc2_stats": {
                "word_count": 120,
                "sentence_count": 6,
                "unique_words": 90,
            },
            "matching_segments_detail": [
                {
                    "text1": "Sample text from doc 1",
                    "text2": "Sample text from doc 2",
                    "score": 0.85,
                    "position1": 0,
                    "position2": 0,
                }
            ],
        }

    def test_generate_json(self):
        """Test JSON report generation"""
        json_report = self.generator.generate_json(self.mock_results)
        
        assert json_report is not None
        assert isinstance(json_report, str)
        assert "overall_similarity" in json_report

    def test_generate_pdf(self):
        """Test PDF report generation"""
        pdf_data = self.generator.generate_pdf(self.mock_results)
        
        assert pdf_data is not None
        assert isinstance(pdf_data, bytes)
        assert len(pdf_data) > 0

    def test_generate_excel(self):
        """Test Excel report generation"""
        excel_data = self.generator.generate_excel(self.mock_results)
        
        assert excel_data is not None
        assert isinstance(excel_data, bytes)
        assert len(excel_data) > 0


class TestIntegration:
    """Integration tests for complete workflow"""

    def setup_method(self):
        """Setup test fixtures"""
        self.file_handler = FileHandler()
        self.text_processor = TextProcessor()
        self.similarity_engine = SimilarityEngine()
        self.report_generator = ReportGenerator()

    def test_complete_workflow(self):
        """Test complete plagiarism detection workflow"""
        # Create mock documents
        doc1_content = b"Python is a high-level programming language. It is widely used for web development and data science."
        doc2_content = b"Python is a popular programming language. Many developers use it for building websites and analyzing data."
        
        file1 = BytesIO(doc1_content)
        file1.name = "doc1.txt"
        
        file2 = BytesIO(doc2_content)
        file2.name = "doc2.txt"
        
        # Extract text
        text1 = self.file_handler.extract_text(file1)
        text2 = self.file_handler.extract_text(file2)
        
        assert text1 is not None
        assert text2 is not None
        
        # Preprocess
        processed1 = self.text_processor.preprocess(text1)
        processed2 = self.text_processor.preprocess(text2)
        
        assert len(processed1) > 0
        assert len(processed2) > 0
        
        # Chunk
        chunks1 = self.text_processor.chunk_text(processed1)
        chunks2 = self.text_processor.chunk_text(processed2)
        
        assert len(chunks1) > 0
        assert len(chunks2) > 0
        
        # Analyze
        results = self.similarity_engine.analyze(
            text1, text2, chunks1, chunks2, threshold=0.7, analysis_mode="Standard"
        )
        
        assert results is not None
        assert "overall_similarity" in results
        assert "risk_level" in results
        
        # Generate reports
        json_report = self.report_generator.generate_json(results)
        pdf_report = self.report_generator.generate_pdf(results)
        excel_report = self.report_generator.generate_excel(results)
        
        assert json_report is not None
        assert pdf_report is not None
        assert excel_report is not None


# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])