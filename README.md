rag/
│
├── 🚀 LIVE FILES (Current Implementation)
│   ├── app.py                         # Interactive web interface
│   ├── requirements.txt               # Python packages needed
│   ├── .env.example                   # Template for environment vars
│   ├── final_legal_embeddings.npy     # Pre-trained vector embeddings
│   ├── final_legal_laws_metadata.json # Legal text with chunk IDs
│   └── README.md                      # Project documentation
│
├── 🔧 PROCESSING SCRIPTS
│   ├── pdf_to_text.py                 # PDF → JSON extraction
│   ├── add_chunk_ids.py               # Add npc2017_* identifiers
│   └── create_embeddings.py           # Generate embeddings
│
└── 📦 ARCHIVE (Previous Versions)
    ├── pdf→text_nochunk/              # Initial text extraction
    ├── embedding/                     # Embedding generation outputs
    └── chunk_id-add/                  # Chunk ID processing
