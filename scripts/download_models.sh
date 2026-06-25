#!/usr/bin/env bash
# Download required models for EHR Copilot
# Embeddings run locally; LLM can use OpenRouter (no download needed)

set -euo pipefail

echo "=== EHR Copilot Model Setup ==="

# Download PubMedBERT embeddings (will be cached by sentence-transformers)
echo "Pre-downloading PubMedBERT embeddings..."
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('NeuML/pubmedbert-base-embeddings')
print(f'Embedding model loaded: dimension={model.get_sentence_embedding_dimension()}')
print('Embeddings model ready!')
"

echo ""
echo "=== Setup Complete ==="
echo "Embeddings: NeuML/pubmedbert-base-embeddings (local)"
echo ""
echo "For LLM, you have two options:"
echo "  1. OpenRouter (recommended): Set EHR__LLM__API_KEY in .env"
echo "  2. Local Ollama: ollama pull qwen2.5:7b-instruct"
