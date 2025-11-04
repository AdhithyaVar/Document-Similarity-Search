"""
Advanced Document Similarity & Plagiarism Detection System
Main Streamlit Application
Version: 2.0.0
"""

import streamlit as st
import time
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config.settings import (
    APP_TITLE,
    APP_ICON,
    SUPPORTED_FORMATS,
    MAX_FILE_SIZE,
    SIMILARITY_THRESHOLDS,
)
from utils.file_handler import FileHandler
from utils.text_processor import TextProcessor
from utils.similarity_engine import SimilarityEngine
from utils.report_generator import ReportGenerator


# Page Configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
def load_custom_css():
    """Load custom CSS styling"""
    css_file = Path(__file__).parent / "assets" / "styles.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_custom_css()


# Initialize Session State
def init_session_state():
    """Initialize session state variables"""
    if "analysis_complete" not in st.session_state:
        st.session_state.analysis_complete = False
    if "results" not in st.session_state:
        st.session_state.results = None
    if "history" not in st.session_state:
        st.session_state.history = []


init_session_state()


# Sidebar Configuration
def render_sidebar():
    """Render sidebar with advanced options"""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("Analysis Settings")
        analysis_mode = st.selectbox(
            "Analysis Depth",
            ["Quick", "Standard", "Deep"],
            index=1,
            help="Quick: Fast scan | Standard: Balanced | Deep: Comprehensive",
        )
        
        chunk_size = st.slider(
            "Chunk Size (tokens)",
            min_value=100,
            max_value=1000,
            value=500,
            step=50,
            help="Larger chunks = faster but less granular",
        )
        
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Minimum score to flag as similar",
        )
        
        st.divider()
        
        st.subheader("Advanced Options")
        show_heatmap = st.checkbox("Show Similarity Heatmap", value=True)
        show_highlights = st.checkbox("Highlight Matching Segments", value=True)
        show_statistics = st.checkbox("Show Detailed Statistics", value=True)
        
        st.divider()
        
        st.subheader("Export Options")
        export_format = st.selectbox(
            "Report Format",
            ["PDF", "Excel", "JSON"],
            help="Choose export format for results",
        )
        
        st.divider()
        
        # System Info
        with st.expander("📊 System Information"):
            st.caption(f"**Version**: 2.0.0")
            st.caption(f"**Model**: all-MiniLM-L6-v2")
            st.caption(f"**Max File Size**: {MAX_FILE_SIZE // (1024*1024)}MB")
            st.caption(f"**Supported Formats**: {', '.join(SUPPORTED_FORMATS)}")
        
        return {
            "analysis_mode": analysis_mode,
            "chunk_size": chunk_size,
            "similarity_threshold": similarity_threshold,
            "show_heatmap": show_heatmap,
            "show_highlights": show_highlights,
            "show_statistics": show_statistics,
            "export_format": export_format,
        }


# Main Header
def render_header():
    """Render main application header"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.title(f"{APP_ICON} {APP_TITLE}")
        st.markdown(
            """
            <div style='text-align: center; color: #666; margin-bottom: 2rem;'>
                Enterprise-grade plagiarism detection with advanced semantic analysis
            </div>
            """,
            unsafe_allow_html=True,
        )


# File Upload Section
def render_upload_section():
    """Render file upload interface"""
    st.subheader("📁 Upload Documents")
    
    col1, col2 = st.columns(2)
    
    with col1:
        file1 = st.file_uploader(
            "Document 1 (Source)",
            type=SUPPORTED_FORMATS,
            help=f"Max size: {MAX_FILE_SIZE // (1024*1024)}MB",
            key="file1",
        )
        if file1:
            st.success(f"✓ {file1.name} ({file1.size / 1024:.1f} KB)")
    
    with col2:
        file2 = st.file_uploader(
            "Document 2 (Comparison)",
            type=SUPPORTED_FORMATS,
            help=f"Max size: {MAX_FILE_SIZE // (1024*1024)}MB",
            key="file2",
        )
        if file2:
            st.success(f"✓ {file2.name} ({file2.size / 1024:.1f} KB)")
    
    return file1, file2


# Analysis Section
def perform_analysis(file1, file2, config):
    """Perform plagiarism analysis"""
    
    # Initialize handlers
    file_handler = FileHandler()
    text_processor = TextProcessor(chunk_size=config["chunk_size"])
    similarity_engine = SimilarityEngine()
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Extract text
        status_text.text("📄 Extracting text from documents...")
        progress_bar.progress(10)
        
        text1 = file_handler.extract_text(file1)
        text2 = file_handler.extract_text(file2)
        
        if not text1 or not text2:
            st.error("❌ Unable to extract text from one or both documents")
            return None
        
        progress_bar.progress(25)
        
        # Step 2: Preprocess
        status_text.text("🔧 Preprocessing and cleaning text...")
        processed1 = text_processor.preprocess(text1)
        processed2 = text_processor.preprocess(text2)
        progress_bar.progress(40)
        
        # Step 3: Chunk documents
        status_text.text("✂️ Chunking documents for analysis...")
        chunks1 = text_processor.chunk_text(processed1)
        chunks2 = text_processor.chunk_text(processed2)
        progress_bar.progress(55)
        
        # Step 4: Compute similarity
        status_text.text("🧠 Computing semantic similarity...")
        results = similarity_engine.analyze(
            text1=text1,
            text2=text2,
            chunks1=chunks1,
            chunks2=chunks2,
            threshold=config["similarity_threshold"],
            analysis_mode=config["analysis_mode"],
        )
        progress_bar.progress(85)
        
        # Step 5: Generate insights
        status_text.text("📊 Generating insights...")
        results["file1_name"] = file1.name
        results["file2_name"] = file2.name
        results["config"] = config
        progress_bar.progress(100)
        
        status_text.text("✅ Analysis complete!")
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()
        
        return results
        
    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        progress_bar.empty()
        status_text.empty()
        return None


# Results Display Section
def display_results(results, config):
    """Display analysis results"""
    
    st.divider()
    st.header("📊 Analysis Results")
    
    # Summary Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        similarity_score = results["overall_similarity"]
        st.metric(
            "Overall Similarity",
            f"{similarity_score:.1%}",
            delta=f"{similarity_score - 0.5:.1%}" if similarity_score > 0.5 else None,
        )
    
    with col2:
        risk_level = results["risk_level"]
        risk_colors = {"High": "🔴", "Moderate": "🟡", "Low": "🟢"}
        st.metric("Risk Level", f"{risk_colors.get(risk_level, '')} {risk_level}")
    
    with col3:
        st.metric("Matching Segments", f"{results['matching_segments']}")
    
    with col4:
        st.metric("Unique Content", f"{results['unique_percentage']:.1%}")
    
    st.divider()
    
    # Detailed Interpretation
    st.subheader("🔍 Detailed Analysis")
    
    interpretation = results.get("interpretation", "")
    if results["overall_similarity"] > 0.8:
        st.error(f"⚠️ **High Similarity Detected**\n\n{interpretation}")
    elif results["overall_similarity"] > 0.5:
        st.warning(f"⚠️ **Moderate Similarity Found**\n\n{interpretation}")
    else:
        st.success(f"✅ **Low Similarity - Mostly Original**\n\n{interpretation}")
    
    # Heatmap
    if config["show_heatmap"] and "heatmap_data" in results:
        st.subheader("🗺️ Similarity Heatmap")
        st.plotly_chart(results["heatmap_data"], use_container_width=True)
    
    # Statistics
    if config["show_statistics"]:
        st.subheader("📈 Statistical Breakdown")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Document 1 Statistics**")
            stats1 = results.get("doc1_stats", {})
            st.write(f"- Word Count: {stats1.get('word_count', 0):,}")
            st.write(f"- Sentence Count: {stats1.get('sentence_count', 0):,}")
            st.write(f"- Unique Words: {stats1.get('unique_words', 0):,}")
        
        with col2:
            st.markdown("**Document 2 Statistics**")
            stats2 = results.get("doc2_stats", {})
            st.write(f"- Word Count: {stats2.get('word_count', 0):,}")
            st.write(f"- Sentence Count: {stats2.get('sentence_count', 0):,}")
            st.write(f"- Unique Words: {stats2.get('unique_words', 0):,}")
    
    # Matching Segments
    if config["show_highlights"] and "matching_segments_detail" in results:
        st.subheader("🎯 Matching Text Segments")
        
        for idx, segment in enumerate(results["matching_segments_detail"][:5], 1):
            with st.expander(f"Match {idx} - Similarity: {segment['score']:.1%}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Document 1:**\n{segment['text1']}")
                with col2:
                    st.markdown(f"**Document 2:**\n{segment['text2']}")


# Export Section
def render_export_section(results, config):
    """Render export options"""
    
    st.divider()
    st.subheader("💾 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    report_gen = ReportGenerator()
    
    with col1:
        if st.button("📄 Generate PDF Report", use_container_width=True):
            with st.spinner("Generating PDF..."):
                pdf_data = report_gen.generate_pdf(results)
                if pdf_data:
                    st.download_button(
                        "⬇️ Download PDF",
                        data=pdf_data,
                        file_name="plagiarism_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
    
    with col2:
        if st.button("📊 Generate Excel Report", use_container_width=True):
            with st.spinner("Generating Excel..."):
                excel_data = report_gen.generate_excel(results)
                if excel_data:
                    st.download_button(
                        "⬇️ Download Excel",
                        data=excel_data,
                        file_name="plagiarism_report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )
    
    with col3:
        if st.button("📋 Copy JSON Data", use_container_width=True):
            json_data = report_gen.generate_json(results)
            st.code(json_data, language="json")


# Main Application Flow
def main():
    """Main application entry point"""
    
    # Render components
    render_header()
    config = render_sidebar()
    file1, file2 = render_upload_section()
    
    st.divider()
    
    # Analysis button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button(
            "🔍 Analyze Documents",
            type="primary",
            use_container_width=True,
            disabled=not (file1 and file2),
        )
    
    # Perform analysis
    if analyze_button and file1 and file2:
        # Validate file sizes
        if file1.size > MAX_FILE_SIZE or file2.size > MAX_FILE_SIZE:
            st.error(f"❌ File size exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit")
            return
        
        results = perform_analysis(file1, file2, config)
        
        if results:
            st.session_state.results = results
            st.session_state.analysis_complete = True
            
            # Add to history
            st.session_state.history.append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "file1": file1.name,
                "file2": file2.name,
                "similarity": results["overall_similarity"],
            })
    
    # Display results if available
    if st.session_state.analysis_complete and st.session_state.results:
        display_results(st.session_state.results, config)
        render_export_section(st.session_state.results, config)
    
    # History section
    if st.session_state.history:
        with st.sidebar:
            st.divider()
            with st.expander("📜 Analysis History"):
                for item in reversed(st.session_state.history[-5:]):
                    st.caption(
                        f"{item['timestamp']}: {item['file1']} vs {item['file2']} "
                        f"({item['similarity']:.1%})"
                    )


# Footer
def render_footer():
    """Render application footer"""
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 2rem 0;'>
            <p>Advanced Plagiarism Detection System v2.0.0</p>
            <p>Powered by Sentence Transformers & Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
    render_footer()