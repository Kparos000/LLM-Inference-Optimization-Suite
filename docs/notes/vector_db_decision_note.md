# Vector Database Decision Note

This project currently supports direct corpus retrieval over generated context JSONL files.

The next retrieval upgrade should evaluate a local vector index such as FAISS or LanceDB so that mm1_dense_top5 becomes real embedding-based retrieval instead of a local fallback.

Recommended future retrieval architecture:
- lexical retrieval
- dense vector retrieval
- metadata-aware hybrid fusion
- deterministic context compression

This note is documentation-only and does not change runtime behavior.
