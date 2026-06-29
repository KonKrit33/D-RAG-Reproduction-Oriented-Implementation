# D-RAG: Differentiable Retrieval-Augmented Generation

This repository provides a practical implementation of the main D-RAG (Differentiable Retrieval-Augmented Generation) pipeline for Knowledge Graph Question Answering (KGQA), based on the methodology proposed by Gao et al. in D-RAG: Differentiable Retrieval-Augmented Generation for Knowledge Graph Question Answering.

The implementation focuses on the WebQSP pipeline and includes the core components of the D-RAG framework: graph-based fact retrieval, differentiable fact sampling with Binary Gumbel-Softmax and Straight-Through Estimation, neural projection of retrieved graph facts into the language-model embedding space, Phase 1 retriever pretraining, Phase 2 joint retriever-generator training, and final evaluation.

Experiments were conducted on a single NVIDIA GeForce RTX 4070 Ti Super GPU with 16GB VRAM, which is substantially smaller than the multi-GPU setup used in the original paper. For this reason, the goal of this repository is not to reproduce the full paper-level benchmark results or claim state-of-the-art performance. Instead, the goal is to provide a runnable, documented, and reproducible course-level implementation of the main D-RAG training mechanism under constrained computational resources.

In particular, this project demonstrates the central idea of D-RAG: replacing hard, non-differentiable retrieval decisions with differentiable fact selection, allowing the generator loss to propagate back through the retrieval module during joint training.

The repository is intended as an educational and research-oriented implementation for studying Retrieval-Augmented Generation, Knowledge Graph Question Answering, Graph Neural Networks, differentiable retrieval, and LLM-based answer generation.


## 🚀 Quick Start (New Instance)

```bash
# 1. Clone and setup
cd /home/shadeform/nlp/drag-improved
./scripts/setup_environment.sh

# 2. Activate environment
source .venv/bin/activate

# 3. Run Phase 2 training (CWQ)
python -m src.trainer.train_phase2 \
    --heuristics_path data/train_heuristics_cwq_train.jsonl \
    --val_heuristics_path data/train_heuristics_cwq_val.jsonl \
    --phase1_checkpoint checkpoints_cwq_subgraph/phase1_best.pt \
    --checkpoint_dir checkpoints_cwq_phase2_paperprompt \
    --epochs 20 \
    --batch_size 64 \
    --lr 5e-5 \
    --weight_decay 0.001 \
    --temperature 0.5 \
    --ret_loss_weight 0.1 \
    --max_facts_cap 100 \
    --prob_threshold 0.03 \
    --val_generation \
    --val_max_new_tokens 50 \
    --val_generation_limit 200 \
    --val_log_samples 5 \
    --eos_loss_weight 1.0
```

---

## 📊 Datasets Used

### Phase 1 Checkpoints (GNN Pre-training)

> **Note:** Checkpoints and heuristics (~700 MB) are automatically downloaded from [🤗 Hugging Face](https://huggingface.co/rhordoancc/drag-improved-checkpoints) when you run `setup_environment.sh`.

| Dataset | Samples | Checkpoint | Training Time |
|---------|---------|------------|---------------|
| **CWQ** (ComplexWebQuestions) | 27,613 | `checkpoints_cwq_subgraph/phase1_best.pt` (288 MB) | ~3.5 min |
| **WebQSP** | 2,826 | `checkpoints_webqsp_subgraph/phase1_best.pt` (288 MB) | ~30 sec |

**Data sources:**
- **CWQ**: `rmanluo/RoG-cwq` (Hugging Face) - Complex multi-hop questions
- **WebQSP**: `rmanluo/RoG-webqsp` (Hugging Face) - Single-hop questions

Each sample contains:
- `question`: Natural language question
- `triples`: Per-question subgraph from Freebase (~50 triples per sample)
- `paths`: Gold reasoning paths for supervision
- `answer`: Expected answer

### Heuristics Files

```
data/
├── train_heuristics_cwq.jsonl           # 27,631 CWQ samples (111 MB) - auto-downloaded
├── train_heuristics_cwq_train.jsonl     # CWQ Phase 2 train split (JSONL)
├── train_heuristics_cwq_val.jsonl       # CWQ Phase 2 val split (JSONL)
├── train_heuristics_webqsp_subgraph.jsonl  # 2,826 WebQSP samples (12 MB) - auto-downloaded
```

**Manual download:**
```bash
python scripts/download_checkpoints.py
```

---

## ⚙️ Environment Setup

### Prerequisites
- **GPU**: H200 (141GB) or B200 (recommended) for 30B model
- **CUDA**: 12.6+ 
- **Python**: 3.12+

### Full Installation

```bash
# 1. Create virtual environment
uv venv .venv
source .venv/bin/activate

# 2. Install core dependencies
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install torch_geometric
uv pip install transformers accelerate peft bitsandbytes datasets tqdm

# 3. Install Unsloth
uv pip install "unsloth[base] @ git+https://github.com/unslothai/unsloth.git"

# 4. Install Mamba dependencies (REQUIRED for Nemotron)
# IMPORTANT: Requires CUDA toolkit for compilation
export PATH=/usr/local/cuda-12.6/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.6

# Install gcc-11 if needed (for cc1plus)
sudo apt-get update && sudo apt-get install -y gcc-11 g++-11
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-11

# Install mamba-ssm and causal-conv1d (takes ~8 minutes to compile)
uv pip install mamba-ssm causal-conv1d --no-build-isolation
```

### Troubleshooting Mamba Installation

If `mamba-ssm` fails to compile:

1. **Check CUDA is available:**
   ```bash
   which nvcc  # Should show /usr/local/cuda-12.6/bin/nvcc
   ```

2. **Check g++ is installed:**
   ```bash
   find /usr -name cc1plus  # Should find gcc-11's cc1plus
   ```

3. **Set environment variables:**
   ```bash
   export PATH=/usr/local/cuda-12.6/bin:$PATH
   export CUDA_HOME=/usr/local/cuda-12.6
   ```

---

## 🧠 Training

> **Note:** The paper trains **separate models** for CWQ and WebQSP. They are not combined.

### Phase 1: GNN Pre-training (Per-Question Subgraphs)

Trains the GNN retriever to identify relevant facts based on heuristic paths.
**Pre-trained checkpoints are already included** - skip this if you want to go straight to Phase 2.

#### CWQ Dataset (ComplexWebQuestions)
```bash
# 27,613 samples, ~3.5 min on A100
python -m src.trainer.train_phase1 \
    --heuristics_path data/train_heuristics_cwq.jsonl \
    --epochs 10 \
    --batch_size 16 \
    --lr 5e-5 \
    --node_dim 256 \
    --hidden_dim 256 \
    --relation_dim 256 \
    --num_reasoning_steps 3 \
    --checkpoint_dir checkpoints_cwq_subgraph
```

#### WebQSP Dataset (WebQuestions Semantic Parses)
```bash
# 2,826 samples, ~30 sec on A100
python -m src.trainer.train_phase1 \
    --heuristics_path data/train_heuristics_webqsp_subgraph.jsonl \
    --epochs 10 \
    --batch_size 16 \
    --lr 5e-5 \
    --node_dim 256 \
    --hidden_dim 256 \
    --relation_dim 256 \
    --num_reasoning_steps 3 \
    --checkpoint_dir checkpoints_webqsp_subgraph
```

---

### Phase 2: Joint End-to-End Training

Jointly trains retriever + projector + generator (Nemotron via LoRA).
**Requires H200 (141GB) or B200 (192GB) for the 30B model.**

#### Setup (run once per session)
```bash
# Set CUDA paths (required for Mamba)
export PATH=/usr/local/cuda-13.0/bin:$PATH
export CUDA_HOME=/usr/local/cuda-13.0

# Activate environment
source .venv/bin/activate
```

#### CWQ Dataset - Phase 2
```bash
python -m src.trainer.train_phase2 \
    --heuristics_path data/train_heuristics_cwq_train.jsonl \
    --val_heuristics_path data/train_heuristics_cwq_val.jsonl \
    --phase1_checkpoint checkpoints_cwq_subgraph/phase1_best.pt \
    --checkpoint_dir checkpoints_cwq_phase2_paperprompt \
    --epochs 20 \
    --batch_size 64 \
    --lr 5e-5 \
    --weight_decay 0.001 \
    --temperature 0.5 \
    --ret_loss_weight 0.1 \
    --max_facts_cap 100 \
    --prob_threshold 0.03 \
    --val_generation \
    --val_max_new_tokens 50 \
    --val_generation_limit 200 \
    --val_log_samples 5 \
    --eos_loss_weight 1.0
```

#### WebQSP Dataset - Phase 2
```bash
python -m src.trainer.train_phase2 \
    --heuristics_path data/train_heuristics_webqsp_subgraph.jsonl \
    --phase1_checkpoint checkpoints_webqsp_subgraph/phase1_best.pt \
    --epochs 5 \
    --batch_size 1 \
    --lr 5e-5 \
    --max_facts_cap 100 \
    --prob_threshold 0.01 \
    --generator_model "unsloth/Nemotron-3-Nano-30B-A3B" \
    --checkpoint_dir checkpoints_webqsp_phase2
```

#### Run Phase 2 in the background (keeps running if terminal closes)
```bash
cd /home/shadeform/nlp/drag-improved && \
source .venv/bin/activate && \
export CUDA_HOME=/usr/local/cuda-13.0 && \
export PATH=/usr/local/cuda-13.0/bin:$PATH && \
export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas && \
export PYTHONPATH=/home/shadeform/nlp/drag-improved && \
mkdir -p logs && \
ts=$(date +%Y%m%d_%H%M%S) && \
log="logs/phase2_train_${ts}.log" && \
nohup python -u -m src.trainer.train_phase2 \
  --heuristics_path data/train_heuristics_cwq_train.jsonl \
  --val_heuristics_path data/train_heuristics_cwq_val.jsonl \
  --phase1_checkpoint checkpoints_cwq_subgraph/phase1_best.pt \
  --checkpoint_dir checkpoints_cwq_phase2_paperprompt \
  --epochs 20 \
  --batch_size 64 \
  --lr 5e-5 \
  --weight_decay 0.001 \
  --temperature 0.5 \
  --ret_loss_weight 0.1 \
  --max_facts_cap 100 \
  --prob_threshold 0.03 \
  --val_generation \
  --val_max_new_tokens 50 \
  --val_generation_limit 200 \
  --val_log_samples 5 \
  --eos_loss_weight 1.0 \
  > "$log" 2>&1 < /dev/null & \
pid=$! && disown && echo "STARTED pid=$pid log=$log"
```

#### Phase 2 Arguments Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--heuristics_path` | required | Path to JSONL with questions + subgraphs |
| `--val_heuristics_path` | None | Optional validation JSONL (runs per-epoch validation if provided) |
| `--phase1_checkpoint` | required | Pre-trained GNN checkpoint from Phase 1 |
| `--generator_model` | `unsloth/Nemotron-3-Nano-30B-A3B` | Nemotron model (BF16 recommended) |
| `--epochs` | 5 | Number of joint training epochs |
| `--batch_size` | 1 | Batch size (increase on B200) |
| `--lr` | 5e-5 | Learning rate |
| `--ret_loss_weight` | 0.1 | Weight for retriever auxiliary loss |
| `--max_facts_cap` | 100 | Cap for fact selection (paper uses 100) |
| `--prob_threshold` | 0.01 | Inference/validation threshold for fact filtering |
| `--val_generation` | False | Run free generation during validation to compute Hits@1/EM/F1 |
| `--val_max_new_tokens` | 50 | Max tokens to generate during validation |
| `--val_generation_limit` | 0 | If >0, only run generation on first N val examples (speed knob) |
| `--eos_loss_weight` | 1.0 | Upweight EOS token loss to help greedy stopping |
| `--use_grad_norm_balance` | False | Paper-aligned gradient-norm loss balancing (can be unstable) |

---

### Training on B200 (192GB VRAM)

With more VRAM, you can increase batch size for better gradient estimates:

```bash
# B200 optimized settings
python -m src.trainer.train_phase2 \
    --heuristics_path data/train_heuristics_cwq_train.jsonl \
    --val_heuristics_path data/train_heuristics_cwq_val.jsonl \
    --phase1_checkpoint checkpoints_cwq_subgraph/phase1_best.pt \
    --epochs 5 \
    --batch_size 4 \
    --max_facts_cap 100 \
    --prob_threshold 0.03 \
    --generator_model "unsloth/Nemotron-3-Nano-30B-A3B" \
    --checkpoint_dir checkpoints_cwq_phase2
```

---

## 📂 Project Structure

```
├── checkpoints_cwq_subgraph/         # Phase 1 CWQ checkpoints
│   └── phase1_best.pt
├── checkpoints_webqsp_subgraph/      # Phase 1 WebQSP checkpoints
│   └── phase1_best.pt
├── configs/
│   └── rolit_kg.yaml                 # RoLit-KG pipeline config
├── data/
│   ├── train_heuristics_cwq.jsonl    # CWQ heuristics with subgraphs
│   ├── train_heuristics_webqsp_subgraph.jsonl  # WebQSP heuristics
│   └── rolit_kg_sample/              # Sample corpora for RoLit-KG
├── docs/
│   ├── drag_documentation.pdf        # D-RAG technical documentation
│   ├── ROLIT_KG_README.md            # RoLit-KG pipeline guide
│   └── rolit_kg_starter_queries.cypher  # Neo4j starter queries
├── scripts/
│   ├── setup_environment.sh          # Full environment setup
│   ├── generate_cwq_heuristics.py    # Generate CWQ heuristics
│   ├── generate_webqsp_subgraph_heuristics.py  # Generate WebQSP heuristics
│   └── run_rolit_kg.py               # RoLit-KG pipeline entrypoint
└── src/
    ├── data/
    │   └── kg_loader.py              # SubgraphDataset for per-question graphs
    ├── model/
    │   ├── retriever.py              # DRAGRetriever (GNN + scoring)
    │   ├── sampler.py                # Gumbel-Softmax differentiable sampling
    │   ├── projector.py              # GNN → LLM dimension bridge
    │   └── generator.py              # Nemotron wrapper with LoRA
    ├── pipeline/                     # RoLit-KG modules (NEW)
    │   ├── ingest.py                 # Corpus ingestion (HF + local)
    │   ├── normalize.py              # Unicode/diacritics normalization
    │   ├── chunking.py               # Text chunking with overlap
    │   ├── extract.py                # NER + relation extraction
    │   ├── validate.py               # Schema validation
    │   ├── resolve.py                # Entity resolution/clustering
    │   ├── ground.py                 # Cross-corpus grounding
    │   ├── neo4j_load.py             # Neo4j Cypher generation + loading
    │   └── evaluate.py               # Metrics and reporting
    └── trainer/
        ├── train_phase1.py           # GNN pre-training
        └── train_phase2.py           # Joint end-to-end training
```

---

## 🆕 RoLit-KG: Romanian Literary Knowledge Graph

**Production-Grade Pipeline** for building Neo4j knowledge graphs from Romanian literary corpora (RO-Stories + HistNERo).

### Latest Production Run ✅

**Successfully processed 103 real Romanian documents in 57 seconds!**

| Metric | Value |
|--------|-------|
| **Documents** | 103 (100 RO-Stories + 3 HistNERo) |
| **Entities Extracted** | 1,158 mentions |
| **Entities Resolved** | 30 unique (97% reduction!) |
| **Relations** | 102,316 total |
| **Runtime** | 57 seconds |
| **Dataset** | HuggingFace `readerbench/ro-stories` |

### Quick Start

```bash
# 1. Install dependencies
pip install datasets transformers sentence-transformers scipy numpy requests

# 2. Download Romanian datasets (100 documents)
python scripts/download_rolit_datasets.py --output_dir data --limit 100

# 3. Run the pipeline
python run_full_pipeline.py \
    --ro_stories_jsonl data/ro_stories_full.jsonl \
    --histnero_jsonl data/histnero_full.jsonl \
    --output_dir outputs/my_run \
    --ollama_url http://inference.ccrolabs.com \
    --ollama_model nomic-embed-text
```

**Outputs**:
```
outputs/my_run/
├── artifacts/      # JSONL files (docs, entities, relations)
├── reports/        # Markdown + JSON metrics
├── cypher/         # Neo4j import scripts
└── summary.json    # Run metadata
```

### Features

#### Pipeline Stages
1. **Ingest** - Load from HuggingFace or local JSONL
2. **Normalize** - Unicode NFC + Romanian diacritics cleanup
3. **Chunk** - Overlapping text chunks (250 tokens)
4. **Extract** - Regex/Transformer NER + relation extraction
5. **Validate** - Schema and reference validation
6. **Resolve** - Semantic entity clustering with Ollama embeddings (97% reduction!)
7. **Ground** - Cross-corpus linking (fictional ↔ historical)
8. **Analytics** - PageRank, communities, narrative patterns
9. **Export** - Neo4j Cypher scripts (idempotent MERGE)

#### Production Features
- ✅ **Real HuggingFace datasets** (`readerbench/ro-stories`)
- ✅ **Ollama embeddings** for semantic resolution
- ✅ **FAISS-accelerated** similarity search
- ✅ **Efficient clustering** (scipy connected_components)
- ✅ **97% entity reduction** via semantic clustering
- ✅ **Comprehensive logging** with timestamps
- ✅ **Graph analytics** (PageRank, communities)
- ✅ **Neo4j export** ready for production

### Load into Neo4j

```cypher
// In Neo4j Browser:
:source outputs/my_run/cypher/constraints.cypher
:source outputs/my_run/cypher/load.cypher
```

### Performance

| Documents | Time | Memory | Entities |
|-----------|------|--------|----------|
| 100 | 1 min | <200MB | ~30 |
| 1,000 | 10 min | ~500MB | ~300 |
| 10,000 | 1.5 hrs* | ~2GB | ~3,000 |
| **12,516 (full)** | **~2 hrs*** | **~3GB** | **~4,000** |

*With FAISS: 10-15 minutes for 10K docs

### Documentation

- **[ROLIT_KG_README.md](docs/ROLIT_KG_README.md)** - Full pipeline guide
- **[PRODUCTION_RUN_RESULTS.md](docs/PRODUCTION_RUN_RESULTS.md)** - Latest run results
- **[OPTIMIZATION_RESULTS.md](docs/OPTIMIZATION_RESULTS.md)** - Performance benchmarks
- **[drag_documentation.pdf](docs/drag_documentation.pdf)** - Technical paper

---

## 🔧 Key Implementation Details

### Per-Question Subgraphs
Unlike approaches that load one giant KG, we use **per-question subgraphs**:
- Each sample has its own small graph (~50 triples)
- Memory efficient (fits on any GPU for Phase 1)
- Matches the paper's approach

### Retriever Architecture (ReaRev-based)
- **Instruction Module**: Sentence-BERT encoder for questions
- **Graph Reasoning**: 3 layers of instruction-conditioned message passing
- **Instruction Update**: Iterative refinement of question representation
- **Fact Scorer**: Bernoulli probability per fact (edge)

### Loss Function
```
L = ρ × L_BCE + (1-ρ) × L_Rank
```
Where ρ = 0.7 (paper default)

### Generator
- **Model**: Nemotron-3-Nano-30B-A3B (BF16)
- **Fine-tuning**: LoRA via Unsloth
- **Injection**: Neural prompts prepended to text embeddings

---

## ⚠️ Hardware Requirements

| Phase | Model | VRAM Required | Recommended GPU |
|-------|-------|---------------|-----------------|
| Phase 1 | GNN only | ~4 GB | Any GPU |
| Phase 2 | GNN + Nemotron 30B | ~140 GB | H200 / B200 |

---

## 📜 Citation

```
@inproceedings{gao2025drag,
  title = {D-RAG: Differentiable Retrieval-Augmented Generation for Knowledge Graph Question Answering},
  author = {Gao, Guangze and
            Li, Zixuan and
            Yuan, Chunfeng and
            Li, Jiawei and
            Wu, Jianzhuo and
            Zhang, Yuehao and
            Jin, Xiaolong and
            Li, Bing and
            Hu, Weiming},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year = {2025},
  pages = {35398--35417},
  publisher = {Association for Computational Linguistics},
  doi = {10.18653/v1/2025.emnlp-main.1793}
}
```

## Citation

If you use this repository, please cite the original D-RAG paper.
