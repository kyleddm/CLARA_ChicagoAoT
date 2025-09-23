## CLARA: Context-aware Language-Augmented Retrieval Anomaly Detection

CLARA is an anomaly detection framework for multimodal/mobile sensor data that combines a FAISS vector store, a contrastive embedding model, and LLM-driven analysis (via Ollama) for explanations and contextual reasoning.

### Features
- Semantic retrieval over sensor patterns using `FAISS`
- Optional contrastive embedding training with PyTorch
- LLM-based explanations, coherence checks, and contextual deviation analysis
- CSV loader for ExtraSensory-style datasets and synthetic data generation
- Feedback loop scaffolding for continual improvement

### Repository Structure
- `clara/`: Python package containing CLARA components
  - `clara_detector.py`: Orchestrates retrieval, augmentation, and detection
  - `faiss_vector_store.py`: FAISS-based vector store with metadata handling
  - `contrastive_embedding_model.py`: Contrastive embedding training utilities
  - `sensor_data_augmenter.py`: Prompt building from sensor data and retrieved context
  - `contextual_deviation_analyzer.py`: LLM-driven contextual deviation scoring
  - `explanation_driven_detector.py`: Explanation-first detection with coherence gating
  - `extrasensory_csv_loader.py`: CSV loader/utilities and synthetic data generator
  - `ollama_llm.py`: Minimal Ollama client wrapper
- `run_anomaly_detection.py`: End-to-end demo pipeline

### Requirements
- Python 3.9+
- System packages: `faiss` (CPU or GPU build)
- Python packages: `numpy`, `pandas`, `faiss-cpu` or `faiss-gpu`, `requests`, `torch`

Install (CPU example):
```bash
pip install numpy pandas requests torch faiss-cpu
```

### Quick Start
1) Start Ollama and pull a model (example):
```bash
ollama serve &
ollama pull llama3.2:1b
```

2) Run the demo (uses defaults in `run_anomaly_detection.py`):
```bash
python run_anomaly_detection.py --skip-training --embedding-dim 64
```

3) Integration test:
```bash
python tests/test_rag_integration.py llama3.2:1b http://localhost:11434
```

### Using CLARA Programmatically
```python
from clara.clara_detector import CLARA

clara = CLARA(vector_store_path=None, llm_model_name="llama3.2:1b", embedding_dim=64)

normal = {"hostID":"u1","activity":"walking","acc_x":0.1,"acc_y":9.8,"acc_z":0.2}
anomaly = {"hostID":"u1","activity":"walking","acc_x":0.6,"acc_y":8.4,"acc_z":0.9}

clara.add_normal_pattern(normal, "Normal walking pattern")
clara.add_anomaly_pattern(anomaly, "behavioral", "Unusual acceleration profile")

result = clara.detect_anomalies(anomaly, use_llm=True)
print(result)
```

### Notes
- Defaults assume Ollama at `http://localhost:11434` and model `llama3.2:1b`.
- If FAISS or GPU is unavailable, use the CPU build (`faiss-cpu`).
- The contrastive model is optional; CLARA falls back to a simple embedding.

### Citation

If you use CLARA in your research, please cite:

```bibtex
@inproceedings{koh2025clara,
  title={Clara: Context-Aware RAG-LLM Framework for Anomaly Detection in Mobile Device Sensors},
  author={Koh, Chan Young and DeMedeiros, Kyle and Hendawi, Abdeltawab},
  booktitle={2025 26th IEEE International Conference on Mobile Data Management (MDM)},
  pages={1--8},
  year={2025},
  organization={IEEE}
}
```

### License
MIT


