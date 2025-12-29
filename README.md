# RAG Project Repository Structure

rag/
│
├── 🚀 app.py
│ └── Main Streamlit RAG application
├── 📦 requirements.txt
│ └── Python dependencies
├── 🔐 .env.example
│ └── Environment variables template
├── 🧠 final_legal_embeddings.npy
│ └── Vector embeddings database
├── 📋 final_legal_laws_metadata.json
│ └── Legal metadata with chunk IDs
├── 📖 README.md
│ └── Project documentation
│
├── 📁 scripts/
│ ├── 01_pdf_extraction.py
│ │ └── Extract PDF → JSON
│ ├── 02_add_chunk_ids.py
│ │ └── Add chunk identifiers
│ └── 03_generate_embeddings.py
│ └── Create embeddings
│
├── 📁 data/
│ ├── penal_code.pdf
│ │ └── Original PDF document
│ ├── structured_laws.json
│ │ └── Parsed JSON structure
│ └── chunked_laws.json
│ └── JSON with chunk IDs
│
└── 📁 archive/
├── pdf→text_nochunk/
│ └── Initial text extraction
├── embedding/
│ └── Embedding generation outputs
└── chunk_id-add/
└── Chunk ID addition outputs
