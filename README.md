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
- `aot/`: Python package containing modified CLARA components fo the Chicago AoT Dataset, as well as data processing and utility functions
  - `sparkml_Desktop.ipynb`: A jupyter notebook used for data processing through spark
  - `aot_csv_loader.py`:The python script to load the AoT data
  - `aot_by_date.py`:The python script to split the 400GB Chicago AoT data into usable chunks
  - `dates.json`: The list of dates to split the AoT data by.  This is later combined into larger files for ingestion
  - `download_datasheets.py`: Used to pull the sensor datasheets (can be used to allow an LLM more information on the sensors used)
  - `prune_aot.ipynb`: The jupyter notebook used to prunt the AoT *without* using spark (longer, but more stable)
  - Other files that match the names of those in the clara folder perform the same functions, just on AoT data.
- `run_anomaly_detection.py`: End-to-end demo pipeline

### Requirements
- Python 3.9+
- System packages: `faiss` (CPU or GPU build)
- Python packages (mandatory): `numpy`, `pandas`, `faiss-cpu` or `faiss-gpu`, `requests`, `torch`, `json`, `pytz`, `argparse`, `datetime`, `os`, `sys`, `time`, `typing`
- Python packages (optionsl): `pyspark`, `matplotlib`, `plotly`, `findspark`, `pickle`, `sklearn`, `csv`

#### Windows and WSL prep (for Windows 11 users)
- enable virtualization in your BIOS
- install Windows optional features:
  - HyperV->HyperV platform
  - Virtual Machine Platform
  - Windows subsystem for Linux
- Install WSL:
  ```pwsh
  wsl --install
  wsl --update
  ```
- Install your desired Linux distribution:
  ```pwsh
  wsl --install Ubuntu-24.04
  ```
- Run WSL: 
  ```pwsh
    wsl #this starts the linux environment
  ```

#### Virtual environment install for a debian-based Linux Distribution (native or wsl)
```bash
sudo apt install python3-pip python3-penv python3-numpy python3-pandas build-essential nvidia-cuda-toolkit
python3 -m venv <path-to-virtual-env-folder>/clara
cd <path-to-clara-venv>
./pip install faiss-gpu-<CUDAVERSION>  #for instance ./pip install fiass-gpu-cu12 for cuda 12+
./pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/<CUDAVERSION>  #for instance, cu128 for cuda 12.8
```

#### Non-virtual Install (CPU example)
```bash
pip install numpy pandas requests torch faiss-cpu
```

### CLARA Quick Start
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
  - make sure you pass through your llama instance if you're using something like WSL with Ollama residing on Windows!M
- If FAISS or GPU is unavailable, use the CPU build (`faiss-cpu`).
- The contrastive model is optional; CLARA falls back to a simple embedding.
- A config file has been generated so you can use that instead of arguments.  All you need is --use-config when you run run_anomaly_detection.py.
  - The config file assumes sym links to the csv file(s) (input) and a the folder where you want your fiass file and log to go (output).  These need to be created as they're not present in the repo.

### AoT-Specific changes
The Chicago Array of things is a massive sensor dataset collecting environmental information in the city of Chicago spanning multiple years (https://github.com/waggle-sensor/waggle/tree/master/data).
Unlike the dataset CLARA was originally used on, The Chicago AoT is a collection of multiple different sensors catagorized by generic columns such as "sensor," "subsystem," "parameter," "value_raw," and "value_hrf."
Because each sensor not only doesn't have its own column (making the data mixed), but contains long names, networking data and both raw information and human readable information, the following changes are made to the dataset:
- All subsystems, node ids, and parameters were truncated where possible to their most significant descriptors.
- The networking data was stripped from the dataset
- Any sensors that provided no meaningful, changing information (i.e. reports of a sensor's id for isntance), were stripped from the dataset
- Because the raw sensor data does not conform to the instruction sheets' operating parameters or unit, they were stripped (if raw data included simple counts, those values were copied to the value_hrf column if they were missing prior to stripping the raw column)
- If performing the above caused a particular sensor to not produce a hrf value for a given timestamp, that row was removed.
- We chose to leave the remaining columns generic, rather than increasing dimensionality (and sparsity) by converting each unique sensor into it's own column.
All of these changes allow the dataset to be significantly reduced in size (assisting with training, embedding generation, and loading into memory)

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
### TODO
- generalize CLARA so any data type can be used
- general code cleanup and documentation

### License
MIT


