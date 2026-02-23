<div align="center">

# 🚀 Responsive Fine-Tuner (RFT)

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.30+-yellow.svg)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**A Human-in-the-Loop Fine-Tuning Platform for Language Models**

*Empower domain experts to improve AI models through interactive feedback — no ML expertise required*

[🚀 Quick Start](#-quick-start) | [📖 Features](#-features) | [💡 Usage Guide](#-usage-guide) | [🤝 Contributing](#-contributing)

</div>

---

## 🌟 Overview

**Responsive Fine-Tuner (RFT)** is an interactive platform that enables domain experts to fine-tune language models through natural feedback loops. Upload your data, label examples interactively, and watch the model improve in real-time — all through an intuitive web interface.

### 🤖 Portfolio Snapshot

- **Core**: Interactive human-in-the-loop (HITL) pipeline for steering Small Language Models (SLMs) via **incremental LoRA** (parameter-efficient fine-tuning).
- **Technical contribution**: A Gradio-based interface that captures real-time user corrections and triggers lightweight updates, minimizing the compute overhead of “online” adaptation.
- **Research focus**: Measuring the **dynamics of adaptation** — how feedback frequency and update strength impact model calibration and catastrophic forgetting (the stability–plasticity dilemma).

### 🎯 Purpose & Scope

This toolkit bridges the gap between domain expertise and machine learning, designed to help you:

- **Democratize AI Development**: Enable subject matter experts to train models without coding
- **Accelerate Model Adaptation**: Fine-tune pre-trained models for specific domains in minutes
- **Maintain Human Oversight**: Keep humans in the loop for critical decision-making
- **Build Trust Through Transparency**: Visualize model confidence and performance metrics in real-time
- **Scale Iteratively**: Start with small datasets and refine incrementally as you gather feedback

**Important**: This is a development and research tool for rapid prototyping and domain adaptation. For production deployments:

- Use comprehensive validation datasets representing real-world distributions
- Conduct thorough testing across diverse scenarios and edge cases
- Implement monitoring and logging for production model behavior
- Follow established MLOps best practices and governance frameworks
- Consider regulatory compliance requirements for your domain

We encourage researchers and practitioners to use this toolkit as a foundation for human-centered AI development, complementing it with rigorous validation protocols and domain expertise.

---

## 🎭 Why Responsive Fine-Tuner?

### ✨ Built for Domain Experts

🎨 **No Code Required**: Drag-and-drop interface designed for subject matter experts, not data scientists

⚡ **Instant Feedback Loop**: See model improvements in real-time as you provide corrections

🔍 **Full Transparency**: Confidence scores, performance charts, and progress tracking keep you informed

🎯 **Domain-Focused**: Optimized for real-world use cases: medical documents, legal review, customer feedback, and more

### 💪 Powered by State-of-the-Art Tech

🚀 **LoRA Efficiency**: Adapts models using Parameter-Efficient Fine-Tuning (PEFT) — train in minutes, not hours

🧠 **Flexible Architecture**: Works with any Hugging Face transformer model (DistilBERT, RoBERTa, GPT-2, etc.)

🎓 **Smart Training**: Adaptive learning rates, intelligent sampling, and reward shaping via TRL

📊 **Production-Ready**: Deployable locally, in Docker containers, or as enterprise-grade applications

---

## ✨ Features

### 🔬 Core Capabilities

#### 📤 Intelligent Data Upload
- **Multi-Format Support**: Upload `.txt` and `.csv` files seamlessly
- **Auto-Encoding Detection**: Automatically detects and handles various text encodings
- **Smart Preprocessing**: Sanitizes text, removes duplicates, and validates data quality
- **Train/Test Splitting**: Automatic stratified splitting with configurable ratios
- **Dataset Health Metrics**: Preview statistics, class distributions, and data quality indicators

#### 🏷️ Interactive Labeling Interface
- **Model-Assisted Labeling**: Get AI predictions with confidence scores for each document
- **One-Click Corrections**: Accept good predictions or correct errors with a single click
- **Context-Rich Display**: View metadata, file sources, and document statistics
- **Progress Tracking**: Visual indicators show labeling velocity and completion status
- **Batch Operations**: Label multiple similar documents efficiently

#### 🎯 Feedback-Driven Fine-Tuning
- **Incremental Learning**: Model updates automatically after each feedback batch
- **LoRA Adapters**: Memory-efficient updates using Low-Rank Adaptation
- **Reward-Based Training**: TRL (Transformer Reinforcement Learning) integration for alignment
- **Adaptive Learning**: Dynamic learning rate adjustment based on dataset size and feedback quality
- **Smart Sampling**: Prioritizes uncertain predictions for maximum learning impact

#### 📊 Performance Analytics
- **Real-Time Metrics**: Track accuracy, F1-score, precision, and recall as you label
- **Loss Visualization**: Monitor training loss trends across fine-tuning iterations
- **Confidence Calibration**: Analyze model confidence vs. actual accuracy
- **Before/After Comparison**: See performance deltas highlighting improvement areas
- **Export Reports**: Download performance summaries and labeled datasets

#### ⚖️ Stability–Plasticity Experiment (Built-In Logger)

RFT can automatically log a lightweight **stability–plasticity** experiment while you use the Gradio app.

- **Stability (hold-out)**: accuracy on a fixed labeled gold set
- **Plasticity (new feedback)**: accuracy on the newly-labeled feedback batch for that cycle

**Files (defaults):**

- Gold set: `data/experiments/stability_plasticity/gold.csv`
- Results CSV: `data/experiments/stability_plasticity/results.csv`

**How to run (manual, 3–5 cycles):**

1. Start the app: `python run_app.py`
2. Upload any dataset (unlabeled is fine)
3. Label until the model retrains (triggered when feedback count reaches the configured batch size)
4. Repeat 3–5 times — after each retrain, RFT appends one row to `results.csv`

**Generate a README-ready table (+ optional plot):**

```bash
python scripts/stability_plasticity_report.py \
  --results data/experiments/stability_plasticity/results.csv \
  --out-html data/experiments/stability_plasticity/plot.html
```

Then paste the printed Markdown table into your README (and optionally link `plot.html`).

**README-ready results block (paste after you run 3–5 cycles):**

- Table: run the script and paste the Markdown output here.
- Graph: generate `plot.html` and link it (or export a PNG screenshot).

```markdown
### Stability–Plasticity Results

| cycle | feedback_samples | stability_accuracy | plasticity_accuracy | learning_rate | timestamp |
|------:|-----------------:|-------------------:|--------------------:|--------------:|----------|
| 1 | 4 |  |  | 1e-4 |  |
| 2 | 4 |  |  | 1e-4 |  |
| 3 | 4 |  |  | 1e-4 |  |

**Plot:** see `data/experiments/stability_plasticity/plot.html`
```

**Notebook option:** you can also generate the same table/plot via [docs/stability_plasticity_report.ipynb](docs/stability_plasticity_report.ipynb).

**Optional configuration:**

- Set `RFT_GOLD_SET_PATH` to point at your own labeled hold-out set.
- Set `RFT_RESULTS_CSV_PATH` to change where results are written.

#### 🧪 Research Notes (Stability vs. Plasticity)

After you run **3–5 feedback cycles**, add a short write-up here (even 5–8 lines is enough). The goal is to document a real observation about the trade-off between fast adaptation and retention.

**What to capture (minimum viable “research artifact”):**

- A small table or plot generated from `data/experiments/stability_plasticity/results.csv`
- 1–2 screenshots of the Gradio labeling flow (before/after a cycle)
- A short paragraph interpreting the trend

**Write-up template (fill in with your numbers):**

> Over **N cycles**, plasticity (new feedback accuracy) improved from **A → B** ($+\Delta$), while stability (gold set accuracy) changed from **C → D** ($\Delta$). In our runs, increasing update strength (e.g., learning rate / batch size) led to **faster adaptation** but **more forgetting** on the gold set, suggesting a clear stability–plasticity trade-off.

#### 🛑 Decision Matrix / When to Stop

- **Stop coding when**: the Gradio demo works end-to-end, `results.csv` has **3–5 rows** (cycles), and you have **1–2 screenshots** showing the loop.
- **Stop writing when**: you have a **single paragraph** summarizing the stability–plasticity trend (even if the trend is modest or mixed).
- **If stability collapses**: reduce `training.learning_rate`, increase the gold set size, or retrain less frequently (larger feedback batches).
- **If plasticity is too slow**: increase feedback batch size slightly, use a slightly higher learning rate, or focus labeling on high-uncertainty samples.

#### 🖼️ Demo Artifacts (Screenshots / GIF)

For a portfolio-ready demo, capture evidence of improvement over **3 feedback cycles**:

- **Screenshots (minimum)**: take 1 screenshot before cycle 1 and 1 screenshot after cycle 3 (same tab, showing prediction + confidence).
- **GIF (optional)**: screen-record the labeling tab for ~30–60 seconds and convert to GIF using your OS screen recorder (or `ffmpeg` if available).

This is intentionally lightweight — the goal is to document the interactive loop, not to build a production-grade MLOps pipeline.

### 🎨 Advanced Features

#### ⚙️ Configurable Training Pipeline
- **Model Selection**: Choose from any Hugging Face model via dropdown or config
- **Hyperparameter Control**: Adjust learning rate, batch size, epochs, and more
- **Auto-Retrain Policies**: Set triggers for automatic fine-tuning (e.g., every N labels)
- **Resource Management**: GPU/CPU selection with memory optimization options

#### 🏢 Enterprise Capabilities
- **Multi-User Support**: Role-based access control ready (`backend/auth.py`)
- **Project Management**: Organize datasets and models into projects
- **Version Control**: Track model versions and dataset snapshots
- **Audit Logging**: Complete history of labels, predictions, and model updates
- **Backup & Recovery**: Automated backups with rollback support (`maintenance/backup.py`)

#### 🔒 Security & Compliance
- **JWT Authentication**: Secure API access with token-based auth
- **Data Encryption**: Protect sensitive datasets during processing
- **Privacy Controls**: Options for on-premise deployment and data isolation
- **Compliance Tools**: Export audit trails for regulatory requirements

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.8 or higher**
- **8GB+ RAM** (16GB recommended for larger models)
- **CUDA-compatible GPU** (optional, but recommended for faster training)
- **Git** for cloning the repository

### Installation (3 Steps)

#### Step 1: Clone the Repository

```bash
git clone https://github.com/dyra-12/Responsive-Fine-Tuner.git
cd Responsive-Fine-Tuner
```

#### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# OR using conda
conda create -n rft python=3.10
conda activate rft
```

#### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### 🎬 Launch the Application

#### Basic Interface

```bash
python run_app.py --port 7860
```

The dashboard will be available at **http://localhost:7860**

#### Advanced Analytics Interface

```bash
python run_advanced_app.py
```

Includes additional features: adaptive training, TRL integration, and enhanced analytics

#### Enterprise Dashboard

```bash
python run_enterprise.py
```

Streamlit-based interface with project management and multi-user features

### 🌐 Public Demo (Share Link)

```bash
python run_app.py --share
```

Generates a public Gradio link for demos and remote collaboration

---

## 💡 Usage Guide

### Quick Workflow (5 Minutes)

1. **Launch App**: Run `python run_app.py`
2. **Upload Data**: Drag & drop your `.txt` or `.csv` files
3. **Start Labeling**: Review predictions and provide corrections
4. **Watch It Learn**: Model automatically improves after each batch
5. **Export Results**: Download labeled data and performance reports

### How the Gradio Feedback Loop Works (Technical)

RFT’s interaction loop is intentionally simple so you can demonstrate “learning from feedback” without heavy infrastructure:

1. **Upload & split**: files are processed and split into train/test.
2. **Predict**: the model predicts a label + confidence for the current item.
3. **Feedback**: you mark the prediction correct/incorrect and supply the correct label when needed.
4. **Buffer**: feedback is stored as a small batch of (text, user_label, prediction).
5. **Incremental update**: when the buffer reaches `training.batch_size`, RFT runs a quick **incremental fine-tune step** using **LoRA** adapters.
6. **Evaluate & log**: after each update, RFT logs stability (gold set) vs plasticity (new batch) to `data/experiments/stability_plasticity/results.csv`.

### Detailed Workflow

#### 🗂️ Step 1: Data Upload

**Tab**: Data Upload

1. Click **"Upload Files"** and select your documents (`.txt` or `.csv`)
2. Review the **Dataset Summary**:
   - Total documents
   - Train/test split
   - Encoding detection results
   - Data quality warnings
3. Check the **Preview Table** showing first few rows
4. Proceed to **Labeling** tab

**💡 Tips**:
- Use sample datasets from `examples/` folder to test the system
- For CSV files, ensure text is in a column named `text` or `content`
- Mixed encodings are handled automatically

**Example Datasets**:
- `examples/sentiment_analysis/sample.csv` — Movie reviews with positive/negative labels
- `examples/medical_document_classification/sample.txt` — Clinical notes for diagnosis classification
- `examples/legal_document_review/sample.csv` — Contract clauses for risk assessment
- `examples/customer_feedback/sample.csv` — Support tickets with urgency labels

---

#### 🏷️ Step 2: Interactive Labeling

**Tab**: Interactive Labeling

1. View the current document and model's prediction
2. Check the **confidence score** (color-coded: green = high, red = low)
3. Choose an action:
   - ✅ **Accept**: If prediction is correct
   - ✏️ **Correct**: Select the right label from dropdown
   - ⏭️ **Skip**: Move to next document
4. Monitor the **Progress Bar** showing completion percentage
5. View **Metadata** for context (file source, timestamp, document ID)

**💡 Tips**:
- Focus on correcting low-confidence predictions first (system highlights them)
- Labeling history is auto-saved — safe to take breaks
- The model retrains automatically after every 10 labels (configurable)

**Visual Indicators**:
- 🟢 **Green badge**: High confidence (>80%)
- 🟡 **Yellow badge**: Medium confidence (50-80%)
- 🔴 **Red badge**: Low confidence (<50%)

---

#### 🎯 Step 3: Model Fine-Tuning

**Tab**: Performance Analytics

Fine-tuning happens automatically in the background, but you can monitor progress:

1. **Training Status**: See current training epoch and loss
2. **Accuracy Chart**: Track model improvement over time
3. **Confusion Matrix**: Understand which classes are confused
4. **Confidence Distribution**: Analyze prediction confidence patterns

**Manual Training** (optional):
```bash
# Trigger training via CLI
python run_advanced_app.py --train-now
```

**💡 Tips**:
- Wait for at least 20-30 labeled examples before evaluating performance
- Loss should decrease over time — if it plateaus, adjust learning rate
- Check confusion matrix to identify problematic class pairs

---

#### 📊 Step 4: Performance Validation

**Tab**: Performance Analytics

**Key Metrics**:
- **Accuracy**: Overall percentage of correct predictions
- **F1-Score**: Balanced metric considering precision and recall
- **Precision**: What % of positive predictions were correct?
- **Recall**: What % of actual positives were found?

**Visualizations**:
- 📈 **Accuracy Timeline**: Performance improvement across labeling sessions
- 🔥 **Loss Curve**: Training loss reduction over epochs
- 📊 **Confidence Histogram**: Distribution of model confidence scores
- 🎯 **Per-Class Performance**: Metrics broken down by label

**💡 Tips**:
- Aim for balanced performance across all classes
- If one class performs poorly, label more examples from that class
- Export metrics periodically to track long-term trends

---

#### 💾 Step 5: Export & Deploy

**Tab**: Settings

**Export Options**:

1. **Labeled Dataset**: Download all labeled examples as CSV
   ```
   document_id, text, label, confidence, timestamp
   ```

2. **Model Adapters**: Save LoRA weights for deployment
   ```bash
   # Saved to: models/rft-model-{timestamp}/
   ```

3. **Performance Report**: Generate PDF summary with charts and metrics

4. **Project Snapshot**: Backup entire project state for reproducibility

**Deployment**:
```bash
# Export for production
python backend/enhanced_model_manager.py --export-model

# Deploy with Docker
docker build -t rft-production -f deployment/Dockerfile .
docker run -p 8080:8080 rft-production
```

---

## 🏗️ Project Structure

```
Responsive-Fine-Tuner/
│
├── run_app.py                   # Main application launcher (basic interface)
├── run_advanced_app.py          # Advanced interface with analytics
├── run_enterprise.py            # Enterprise Streamlit dashboard
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── LICENSE                      # MIT License
├── setup.py                     # Package configuration for pip install
│
├── frontend/                    # User interface components
│   ├── main_app.py              # Gradio app definition with tabs
│   ├── app_core.py              # Core application logic (RFTApplication)
│   ├── enterprise_interface.py  # Streamlit enterprise UI
│   └── components/              # Modular UI components
│       ├── data_upload.py       # File upload and preview
│       ├── labeling.py          # Interactive labeling interface
│       ├── performance.py       # Metrics and charts
│       └── analytics.py         # Advanced analytics dashboards
│
├── backend/                     # Core ML and data processing
│   ├── config.py                # Configuration management (YAML)
│   ├── data_processor.py        # Data ingestion, cleaning, splitting
│   ├── model_manager.py         # Basic model loading and inference
│   ├── enhanced_model_manager.py # LoRA fine-tuning and metrics
│   ├── advanced_trainer.py      # Adaptive training and TRL integration
│   ├── project_manager.py       # Multi-project organization
│   ├── auth.py                  # JWT authentication
│   ├── security.py              # Encryption and access control
│   ├── analytics.py             # Performance metrics and reporting
│   └── optimizations.py         # Memory and compute optimizations
│
├── config/                      # Configuration files
│   ├── settings.yaml            # Default settings (model, training params)
│   └── enterprise.yaml          # Enterprise configuration overrides
│
├── examples/                    # Sample datasets for quick testing
│   ├── README.md                # Guide to example datasets
│   ├── sentiment_analysis/      # Movie reviews (IMDB-style)
│   ├── medical_document_classification/ # Clinical notes
│   ├── legal_document_review/   # Contract clauses
│   └── customer_feedback/       # Support tickets
│
├── docs/                        # Documentation
│   ├── overview.md              # High-level architecture
│   ├── quick_start.md           # 5-minute getting started
│   ├── Tutorial.ipynb           # Jupyter notebook walkthrough
│   └── api/                     # Sphinx API documentation
│       ├── index.rst
│       ├── api_reference.rst
│       └── requirements-docs.txt
│
├── tests/                       # Test suites
│   ├── test_phase1.py           # Data processing tests
│   ├── test_phase2.py           # Model loading tests
│   ├── test_phase3.py           # Labeling workflow tests
│   ├── test_phase4.py           # Training and fine-tuning tests
│   ├── test_phase5.py           # Analytics and metrics tests
│   └── test_phase6.py           # Integration tests
│
├── deployment/                  # Production deployment
│   ├── Dockerfile               # Standard Docker image
│   ├── docker-compose.yml       # Multi-container setup
│   ├── enterprise-docker-compose.yml # Enterprise stack
│   ├── nginx.conf               # Reverse proxy configuration
│   └── production.py            # Production server with monitoring
│
├── maintenance/                 # Operational tools
│   └── backup.py                # Backup and restore utilities
│
├── monitoring/                  # Observability
│   └── monitor.py               # Metrics collection and alerting
│
└── scripts/                     # Automation scripts
    ├── build_and_push_docker.sh # Docker build pipeline
    └── publish_pypi.sh          # PyPI publishing workflow
```

---

## 🧠 Technical Details

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Data Upload │  │   Labeling   │  │ Performance  │         │
│  │  Component   │  │  Component   │  │  Component   │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│         │                  │                  │                  │
│         └──────────────────┼──────────────────┘                 │
│                            │                                     │
└────────────────────────────┼─────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Application Core                           │
│                    (RFTApplication)                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  • State Management (current data, labeling history)    │   │
│  │  • Workflow Orchestration (upload → label → train)      │   │
│  │  • Event Handlers (button clicks, file uploads)         │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│   Data Processor         │  │  Enhanced Model Manager  │
│                          │  │                          │
│  • Encoding Detection    │  │  • Model Loading         │
│  • Text Sanitization     │  │  • LoRA Configuration    │
│  • Train/Test Splitting  │  │  • Incremental Training  │
│  • Quality Validation    │  │  • Metrics Tracking      │
└──────────────────────────┘  └──────────┬───────────────┘
                                         │
                                         ▼
                              ┌──────────────────────┐
                              │  Advanced Trainer    │
                              │                      │
                              │  • Adaptive LR       │
                              │  • Smart Sampling    │
                              │  • TRL Integration   │
                              └──────────────────────┘
```

### Key Technologies

#### 🔧 Core ML Stack

**PyTorch (≥2.0.0)**
- Deep learning framework for model training and inference
- GPU acceleration with CUDA support
- Dynamic computation graphs for flexibility

**Transformers (≥4.30.0)**
- Hugging Face library for pre-trained models
- Access to thousands of models: BERT, RoBERTa, GPT-2, T5, etc.
- Unified API for different architectures

**PEFT (≥0.4.0)**
- Parameter-Efficient Fine-Tuning library
- LoRA (Low-Rank Adaptation) implementation
- Reduces trainable parameters by 90%+ while maintaining performance

**TRL (≥0.4.7)**
- Transformer Reinforcement Learning
- Reward-based alignment and preference learning
- Human feedback integration

#### 🎨 Frontend & UI

**Gradio (≥4.0.0)**
- Web interface with zero frontend code
- Real-time updates and interactivity
- Automatic API endpoint generation

**Plotly (≥5.0.0)**
- Interactive charts and visualizations
- Responsive and mobile-friendly
- Export to PNG/SVG

#### 📊 Data & Analytics

**Pandas (≥1.3.0)**
- Data manipulation and analysis
- CSV/Excel reading with encoding detection

**Scikit-learn (≥1.0.0)**
- Train/test splitting
- Evaluation metrics (accuracy, F1, precision, recall)
- Confusion matrices

**NumPy (≥1.21.0)**
- Numerical computations
- Array operations

### Fine-Tuning Methodology

#### LoRA (Low-Rank Adaptation)

RFT uses LoRA to fine-tune models efficiently:

**How it works**:
1. **Freeze Base Model**: Original weights remain unchanged
2. **Add Low-Rank Matrices**: Inject small trainable matrices (rank = 8-16)
3. **Train Only Adapters**: Update 0.1-1% of parameters
4. **Merge for Inference**: Combine adapters with base model at runtime

**Advantages**:
- ⚡ **Fast**: Train in minutes instead of hours
- 💾 **Memory Efficient**: Requires minimal GPU memory
- 💰 **Cost-Effective**: Lower compute costs
- 🔄 **Reversible**: Easy to swap or merge adapters

**Configuration**:
```python
lora_config = LoraConfig(
    r=8,                    # Rank of adapter matrices
    lora_alpha=32,          # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    lora_dropout=0.1,       # Regularization
    bias="none",
    task_type="SEQ_CLS"     # Sequence classification
)
```

#### Adaptive Training

**Smart Learning Rate Scheduling**:
```python
# Adjusts based on dataset size
if num_samples < 100:
    lr = 5e-4  # Higher LR for small datasets
elif num_samples < 1000:
    lr = 2e-4  # Medium LR
else:
    lr = 1e-4  # Conservative LR for large datasets
```

**Intelligent Sampling**:
- Prioritizes uncertain predictions (confidence < 0.6)
- Balances class distribution
- Avoids overfitting on easy examples

**Reward-Based Learning** (via TRL):
- Incorporates human preferences
- Aligns model behavior with user corrections
- Reduces prediction drift

---

## 🎓 Use Cases

### Research

#### 🔬 Interpretability Studies
- Analyze how models improve with human feedback
- Study the impact of different labeling strategies
- Compare LoRA vs full fine-tuning efficiency

#### 📊 Benchmark Experiments
- Test various pre-trained models on domain-specific tasks
- Evaluate few-shot vs many-shot learning
- Measure human-in-the-loop effectiveness

#### 🧪 Prototyping
- Rapid experimentation with new architectures
- Quick validation of hypotheses
- Iterate on model ideas without heavy infrastructure

### Industry

#### 🏥 Healthcare
- **Medical Document Classification**: Triage clinical notes by urgency
- **Diagnosis Coding**: Auto-assign ICD codes with physician review
- **Adverse Event Detection**: Flag safety reports for review

#### ⚖️ Legal
- **Contract Review**: Identify risky clauses in agreements
- **Legal Document Classification**: Categorize by document type
- **Compliance Monitoring**: Flag non-compliant language

#### 💼 Business
- **Customer Feedback Analysis**: Classify support tickets by topic/urgency
- **Sentiment Analysis**: Monitor brand perception in reviews
- **Email Routing**: Auto-assign emails to correct department

#### 🏦 Finance
- **Transaction Categorization**: Classify expenses automatically
- **Fraud Detection**: Flag suspicious transactions for review
- **Regulatory Compliance**: Identify reportable events

### Education

#### 👩‍🏫 Teaching Tool
- Demonstrate active learning concepts
- Show transformer fine-tuning in action
- Visualize model improvement over time

#### 🎓 Student Projects
- Capstone projects with real-world impact
- Hands-on ML course assignments
- Research paper implementations

#### 🔬 Research Training
- Teach best practices in model evaluation
- Introduce human-in-the-loop workflows
- Practice responsible AI development

---

## 📦 Dependencies

### Core Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| **PyTorch** | ≥2.0.0 | Deep learning framework |
| **Transformers** | ≥4.30.0 | Pre-trained models from Hugging Face |
| **PEFT** | ≥0.4.0 | Parameter-efficient fine-tuning (LoRA) |
| **TRL** | ≥0.4.7 | Reinforcement learning for transformers |
| **Gradio** | ≥4.0.0 | Web interface framework |
| **Datasets** | ≥2.12.0 | Hugging Face dataset utilities |
| **Accelerate** | ≥0.20.0 | Multi-GPU training and optimization |

### Supporting Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| **Pandas** | ≥1.3.0 | Data manipulation and CSV handling |
| **NumPy** | ≥1.21.0 | Numerical computations |
| **Scikit-learn** | ≥1.0.0 | Metrics and data splitting |
| **Plotly** | ≥5.0.0 | Interactive visualizations |
| **PyYAML** | ≥6.0 | Configuration file parsing |
| **PyJWT** | ≥2.6.0 | Authentication tokens |
| **python-dotenv** | ≥0.19.0 | Environment variable management |
| **tqdm** | ≥4.64.0 | Progress bars |

### Optional (for advanced features)

| Library | Purpose |
|---------|---------|
| **bitsandbytes** | 8-bit quantization for memory savings |
| **tensorboard** | Training visualization |
| **wandb** | Experiment tracking |
| **streamlit** | Alternative UI for enterprise features |

See `requirements.txt` for complete list with exact versions.

---

## 🐳 Docker Installation

### Quick Start with Docker

```bash
# Build the image
docker build -t responsive-fine-tuner:latest -f deployment/Dockerfile .

# Run the container
docker run -p 7860:7860 responsive-fine-tuner:latest
```

Access at **http://localhost:7860**

### Docker Compose (with GPU support)

```bash
# Start all services
docker-compose -f deployment/docker-compose.yml up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Enterprise Stack

```bash
# Full stack with monitoring, authentication, and database
docker-compose -f deployment/enterprise-docker-compose.yml up -d
```

**Includes**:
- Gradio frontend on port 7860
- Streamlit enterprise UI on port 8501
- PostgreSQL database for project storage
- Nginx reverse proxy with SSL
- Prometheus + Grafana monitoring

---

## ⚙️ Configuration

### Basic Configuration (`config/settings.yaml`)

```yaml
model:
  name: "distilbert-base-uncased"  # Hugging Face model ID
  max_length: 512                  # Maximum sequence length
  num_labels: 2                    # Number of classes

training:
  batch_size: 8                    # Training batch size
  learning_rate: 2e-4              # Initial learning rate
  num_epochs: 3                    # Epochs per fine-tuning round
  weight_decay: 0.01               # L2 regularization
  warmup_steps: 100                # LR warmup

lora:
  r: 8                             # LoRA rank
  lora_alpha: 32                   # LoRA scaling
  lora_dropout: 0.1                # LoRA dropout
  target_modules: ["q_proj", "v_proj"]

data:
  train_split: 0.8                 # Train/test ratio
  max_samples: 10000               # Limit total samples
  auto_retrain_threshold: 10       # Labels before auto-retrain

ui:
  theme: "default"                 # Gradio theme
  show_confidence: true            # Display confidence scores
  enable_analytics: true           # Show performance charts
```

### Enterprise Configuration (`config/enterprise.yaml`)

```yaml
authentication:
  enabled: true
  jwt_secret: "${JWT_SECRET}"      # From environment
  token_expiry: 86400              # 24 hours

projects:
  max_per_user: 10
  auto_backup: true
  backup_interval: 3600            # 1 hour

monitoring:
  enabled: true
  log_level: "INFO"
  metrics_port: 9090
```

### Environment Variables

Create a `.env` file:

```bash
# Model settings
MODEL_NAME=distilbert-base-uncased
MAX_LENGTH=512

# Authentication
JWT_SECRET=your-secret-key-here
AUTH_ENABLED=true

# Paths
DATA_DIR=./data
MODEL_CACHE_DIR=./models

# Compute
CUDA_VISIBLE_DEVICES=0           # GPU selection
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

---

## 🧪 Testing & Quality Assurance

### Run All Tests

```bash
# Complete test suite
python run_final_tests.py

# Or using pytest
pytest tests/ -v
```

### Phase-Specific Tests

```bash
# Phase 1: Data processing
pytest tests/test_phase1.py -v

# Phase 2: Model loading
pytest tests/test_phase2.py -v

# Phase 3: Labeling workflow
pytest tests/test_phase3.py -v

# Phase 4: Fine-tuning
pytest tests/test_phase4.py -v

# Phase 5: Analytics
pytest tests/test_phase5.py -v

# Phase 6: Integration
pytest tests/test_phase6.py -v
```

### Run Specific Test

```bash
# Test a specific function
pytest tests/test_phase3.py::test_labeling_workflow -v

# Test with keyword matching
pytest tests/ -k "upload" -v
```

### Coverage Report

```bash
pytest tests/ --cov=backend --cov=frontend --cov-report=html
open htmlcov/index.html
```

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### Ways to Contribute

- 🐛 **Bug Reports**: Open an issue with detailed reproduction steps
- ✨ **Feature Requests**: Suggest new capabilities or improvements
- 📝 **Documentation**: Improve guides, add examples, fix typos
- 💻 **Code**: Submit pull requests for new features or bug fixes
- 🎨 **UI/UX**: Enhance the dashboard design and user experience
- 🧪 **Testing**: Add test cases or improve test coverage
- 🌍 **Localization**: Translate UI and docs to other languages

### Development Setup

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/YOUR-USERNAME/Responsive-Fine-Tuner.git
cd Responsive-Fine-Tuner

# Create a feature branch
git checkout -b feature/your-feature-name

# Install dependencies including dev tools
pip install -r requirements.txt
pip install pytest pytest-cov black flake8

# Make your changes
# ...

# Run tests
pytest tests/ -v

# Run linter
flake8 backend/ frontend/ tests/

# Format code
black backend/ frontend/ tests/

# Commit and push
git add .
git commit -m "Add: your feature description"
git push origin feature/your-feature-name

# Open a pull request on GitHub
```

### Code Style Guidelines

- **Follow PEP 8**: Use `black` for auto-formatting
- **Add Docstrings**: Document all functions, classes, and modules
- **Type Hints**: Include type annotations where applicable
- **Write Tests**: Add unit tests for new features
- **Update Docs**: Modify README or docs if user-facing changes

### Pull Request Process

1. **Describe Changes**: Clearly explain what and why in PR description
2. **Link Issues**: Reference related issue numbers (e.g., "Fixes #42")
3. **Pass CI**: Ensure all tests pass in GitHub Actions
4. **Request Review**: Tag maintainers for review
5. **Address Feedback**: Respond to review comments promptly

### Commit Message Format

```
<type>: <short summary>

<optional detailed description>

<optional footer>
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

**Examples**:
```
feat: Add support for multi-label classification
fix: Resolve encoding error in CSV upload
docs: Update installation instructions for Windows
```

---

## 🔭 Future Work / Research Directions

This project currently focuses on **incremental supervised fine-tuning via LoRA** (parameter-efficient updates from human corrections). A natural next step is to transition from “correct label” supervision to **preference-based learning** that better captures nuanced human feedback.

- **Preference aggregation**: collect pairwise preferences (A vs. B) and explore aggregation strategies (majority vote, confidence-weighted votes, disagreement tracking).
- **Reward modeling**: train a reward model on expert preferences to predict which outputs are preferred (and under what conditions).
- **Preference-based optimization (TRL)**: investigate using TRL to optimize the base model against the learned reward signal, enabling richer feedback loops than label-only updates.

This framing positions incremental LoRA as the practical precursor to full **preference-based reward learning**, while keeping the current interactive HITL workflow lightweight and reproducible.

## 📚 Additional Resources

- 📘 **[Quick Start Guide](docs/quick_start.md)** — Get started in 5 minutes
- 📓 **[Tutorial Notebook](docs/Tutorial.ipynb)** — Step-by-step walkthrough with code
- 📖 **[API Documentation](docs/api/index.rst)** — Complete API reference (build with Sphinx)
- 🗺️ **[Roadmap](Roadmap.md)** — Upcoming features and project vision
- 🤝 **[Contributing Guide](CONTRIBUTING.md)** — Detailed contribution guidelines
- 📜 **[Code of Conduct](CODE_OF_CONDUCT.md)** — Community standards
- 📝 **[Changelog](CHANGELOG.md)** — Version history and release notes
- 🧪 **[Example Datasets](examples/README.md)** — Curated samples for testing

### External Resources

- 🤗 **[Hugging Face Transformers Docs](https://huggingface.co/docs/transformers)** — Model library documentation
- 🎓 **[PEFT Tutorial](https://huggingface.co/docs/peft)** — Parameter-efficient fine-tuning guide
- 📊 **[TRL Documentation](https://huggingface.co/docs/trl)** — Reinforcement learning for LLMs
- 🎨 **[Gradio Documentation](https://gradio.app/docs)** — Build ML web interfaces

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Responsive Fine-Tuner Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 💬 Contact & Support

### Get Help

- 📮 **GitHub Issues**: [Report bugs or request features](https://github.com/dyra-12/Responsive-Fine-Tuner/issues)
- 💬 **GitHub Discussions**: [Ask questions or share ideas](https://github.com/dyra-12/Responsive-Fine-Tuner/discussions)
- 📧 **Email**: dyra12@example.com
- 🐦 **Twitter**: [@dyra12](https://twitter.com/dyra12)

### Acknowledgments

Built with ❤️ by the community

**Special Thanks To**:
- Hugging Face team for Transformers and PEFT libraries
- Gradio team for the amazing UI framework
- PyTorch community for the deep learning foundation
- All contributors who have helped improve this project

---

<div align="center">

### ⭐ Star us on GitHub if you find this project useful!

[⬆ Back to Top](#-responsive-fine-tuner-rft)

</div>
