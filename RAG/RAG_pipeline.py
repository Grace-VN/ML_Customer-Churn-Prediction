"""
Semantic RAG Report Generation — Groq + Local Embeddings
- Embeddings: sentence-transformers (all-MiniLM-L6-v2) - LOCAL, FREE, NO API
- Generation: Groq llama-3.3-70b-versatile (free tier)
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# ==============================
# Configuration
# ==============================
load_dotenv()

ROOT_DIR   = Path(__file__).resolve().parents[1]
CSV_DIR    = ROOT_DIR / "output_storage" / "csv_files"
OUTPUT_DIR = ROOT_DIR / "output_storage" / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# API keys
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
GROQ_MODEL     = "llama-3.3-70b-versatile"   # free tier
EMBED_MODEL    = "all-MiniLM-L6-v2"          # local, downloads once
MAX_TOKENS     = 4096

# Validate API key
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found in .env file")

# Initialize clients
groq_client = Groq(api_key=GROQ_API_KEY)

print(f"✓ Initialized Groq client ({GROQ_MODEL})")
print(f"✓ Initializing embedding model ({EMBED_MODEL})...")

# Download embedding model on first run (cached locally after)
embedder = SentenceTransformer(EMBED_MODEL)

print(f"✓ Embedding model ready (local, no API needed)")


# ==============================
# Data Loading
# ==============================
def load_and_chunk_csv_data(csv_dir: Path) -> list:
    """Load CSVs and break into semantic chunks."""
    chunks = []

    csv_files = {
        'classification_report':       ('Classification Report',  'classification_report.csv'),
        'confusion_matrix_components': ('Confusion Matrix',       'confusion_matrix_components.csv'),
        'cost_benefit_analysis':       ('Cost-Benefit Analysis',  'cost_benefit_analysis.csv'),
        'threshold_optimization':      ('Threshold Optimization', 'threshold_optimization.csv'),
        'summary_metrics':             ('Summary Metrics',        'summary_metrics.csv'),
    }

    for key, (section_name, filename) in csv_files.items():
        filepath = csv_dir / filename

        if not filepath.exists():
            print(f"⚠️  Missing: {filename}")
            continue

        df = pd.read_csv(filepath)

        for _, row in df.iterrows():
            chunk_text = f"{section_name}: "
            for col, val in row.items():
                if isinstance(val, (int, float)):
                    chunk_text += f"{col}: {val:,.4f}. "
                else:
                    chunk_text += f"{col}: {val}. "

            chunks.append({
                'text':     chunk_text,
                'section':  section_name,
                'source':   filename,
                'full_row': row.to_dict()
            })

        print(f"✓ {section_name}: {len(df)} chunks")

    return chunks


# ==============================
# Embedding (local, free, no API)
# ==============================
def embed_chunks(chunks: list) -> list:
    """Embed chunks locally using sentence-transformers (no API cost)."""
    print(f"\n🚀 Embedding {len(chunks)} chunks locally ({EMBED_MODEL})...")

    texts      = [c['text'] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    for chunk, emb in zip(chunks, embeddings):
        chunk['embedding'] = emb

    print(f"✓ Embedded {len(chunks)} chunks (all cached locally)")
    return chunks


# ==============================
# Semantic Retrieval
# ==============================
def retrieve_relevant_chunks(query: str, chunks: list, top_k: int = 5) -> list:
    """Return top-k chunks most relevant to query via cosine similarity."""
    
    # Embed query locally (no API call)
    q_vec = embedder.encode([query], convert_to_numpy=True)[0]

    scores = []
    for chunk in chunks:
        d_vec = chunk['embedding']
        
        # Cosine similarity
        norm_q = np.linalg.norm(q_vec)
        norm_d = np.linalg.norm(d_vec)
        
        if norm_q > 0 and norm_d > 0:
            sim = np.dot(q_vec, d_vec) / (norm_q * norm_d)
        else:
            sim = 0
        
        scores.append(sim)

    top_idx = np.argsort(scores)[-top_k:][::-1]
    return [chunks[i] for i in top_idx]


# ==============================
# Context Building
# ==============================
def build_semantic_context(chunks: list) -> dict:
    """Build per-section context via semantic retrieval."""
    section_queries = {
        'executive_summary': "Overall business impact, cost savings, and ROI of the model",
        'performance':       "Accuracy, precision, recall, F1 score, and AUC metrics",
        'cost_benefit':      "Cost breakdown and money saved by the model",
        'risk_analysis':     "False negatives, false positives, and why catching churners matters",
        'thresholds':        "Optimal decision thresholds and precision-recall trade-offs",
        'recommendations':   "Deployment recommendations and monitoring strategy",
    }

    context = {}
    print("\n🔍 Retrieving relevant metrics via semantic search...")

    for section, query in section_queries.items():
        retrieved = retrieve_relevant_chunks(query, chunks, top_k=5)

        lines  = f"## {section.replace('_', ' ').title()}\n"
        lines += f"Query: {query}\n\nRelevant data:\n"
        for i, chunk in enumerate(retrieved, 1):
            lines += f"{i}. {chunk['text']}\n"

        context[section] = lines
        print(f"  ✓ {section}: {len(retrieved)} chunks retrieved")

    return context


# ==============================
# Report Generation (Groq llama-3)
# ==============================
def generate_report(semantic_context: dict) -> str:
    """Generate the business report using Groq llama-3."""
    context_str = "\n\n".join(semantic_context.values())

    system_prompt = (
        "You are an expert data scientist writing a professional business report "
        "for a churn prediction model. Use clear language suitable for executives "
        "and product leaders. Reference specific numbers from the provided data. "
        "Format output in professional markdown."
    )

    user_prompt = f"""Write a comprehensive business report for a customer churn prediction model.

Use this semantically-retrieved data to inform each section:

{context_str}

Structure:
1. Executive Summary (150-200 words, lead with business impact)
2. Model Performance Overview
3. Cost-Benefit Analysis & ROI
4. Risk Assessment & Trade-offs
5. Key Insights
6. Deployment Recommendations
7. Monitoring & Next Steps

Requirements:
- Use specific numbers from the data above
- Write for business audience (executives, risk teams)
- Length: 2000-2500 words
- Professional markdown formatting"""

    print(f"\n✍️  Generating report with Groq ({GROQ_MODEL})...")

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
    )

    return response.choices[0].message.content


# ==============================
# Save Output
# ==============================
def save_report(text: str, output_dir: Path) -> Path:
    """Save report as markdown file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path      = output_dir / f"Churn_Report_LocalRAG_{timestamp}.md"
    path.write_text(text, encoding="utf-8")
    return path


# ==============================
# Main Pipeline
# ==============================
def main():
    print("\n" + "="*70)
    print("  SEMANTIC RAG — Groq + Local Embeddings")
    print("="*70)
    print(f"  Embedding model  : {EMBED_MODEL}  (local, FREE, no API)")
    print(f"  LLM model        : {GROQ_MODEL}   (Groq free tier)")
    print(f"  CSV directory    : {CSV_DIR}")
    print(f"  Total cost       : $0")
    print("="*70)

    # Step 1: Load data
    print("\n📊 Step 1: Loading CSV data...")
    chunks = load_and_chunk_csv_data(CSV_DIR)
    
    if not chunks:
        print("\n❌ No CSV data found. Check CSV_DIR path.")
        return
    
    print(f"\n   Total chunks created: {len(chunks)}")

    # Step 2: Embed chunks (local, no API)
    print("\n📝 Step 2: Creating semantic embeddings (locally)...")
    chunks = embed_chunks(chunks)

    # Step 3: Build semantic context
    print("\n🔎 Step 3: Building semantic context via RAG...")
    context = build_semantic_context(chunks)

    # Step 4: Generate report
    print("\n✨ Step 4: Generating report with Groq...")
    report = generate_report(context)

    # Step 5: Save
    print("\n💾 Step 5: Saving report...")
    path = save_report(report, OUTPUT_DIR)
    print(f"   ✓ Report saved: {path.name}")

    print("\n" + "="*70)
    print("✓ SEMANTIC RAG GENERATION COMPLETE")
    print("="*70)
    print(f"\n📄 Output file: {path}")
    print(f"\n📊 Report preview (first 800 characters):")
    print("-" * 70)
    print(report[:800] + "\n...")
    print("-" * 70)
    
    return path


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\n❌ {type(e).__name__}: {e}")
        print("\n🔧 Troubleshooting:")
        print("  1. Check .env has GROQ_API_KEY")
        print("  2. Verify CSV files exist in CSV_DIR")
        print("  3. Install: pip install groq sentence-transformers python-dotenv numpy pandas")
        print("  4. Internet needed ONLY for first embedder.encode() (downloads model once)")
        print(f"\nNote: First run downloads embedding model (~27MB)")
        traceback.print_exc()