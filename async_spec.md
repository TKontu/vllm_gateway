# Async Processing Configuration Instructions

## Overview

Async batching allows processing multiple chunks concurrently for 3-5x speedup. This requires proper configuration on vLLM server

### **Required vLLM Configuration for Async:**

```bash
# Updated startup command with concurrent support
python -m vllm.entrypoints.openai.api_server \
    --model gemma3-4B \
    --host 0.0.0.0 \
    --port 9003 \
    --max-model-len 4096 \
    --max-num-seqs 8 \                    # CRITICAL: Allow 8 concurrent sequences
    --max-parallel-loading-workers 4 \    # Parallel model loading
    --disable-log-requests \               # Reduce log spam (optional)
    --tensor-parallel-size 1              # Single GPU (adjust if multi-GPU)
```

### **Key Parameters:**

- **`--max-num-seqs 8`**: **CRITICAL** - Without this, vLLM processes requests sequentially
- **`--max-parallel-loading-workers 4`**: Improves startup time for concurrent requests
- **`--disable-log-requests`**: Optional - reduces log volume during high concurrency

### **Memory Considerations (RTX 3090 24GB):**

- **Model size**: ~8GB VRAM for gemma3-4B
- **Concurrent sequences**: ~2-4GB additional VRAM for 8 concurrent requests
- **Total usage**: ~12GB VRAM (leaves 12GB free)
- **Safe concurrency**: 8 sequences is optimal for 24GB VRAM
