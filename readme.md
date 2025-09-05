# Health AI Analysis System

A comprehensive multi-agent AI system for health data analysis with ablation studies and single LLM comparisons.

## Features

- **Multi-Agent Analysis**: Specialized agents for clinical interpretation, anomaly detection, personalized recommendations, and trend analysis
- **Single LLM Comparison**: Comprehensive single LLM implementation for comparison studies
- **Ablation Studies**: Systematic evaluation of individual agent contributions
- **Medical Accuracy**: Evidence-based analysis grounded in clinical research
- **Comprehensive Evaluation**: Quality metrics including completeness, clinical accuracy, coherence, and actionability

## Installation

```bash
git clone <repository-url>
cd health-ai-analysis
pip install -r requirements.txt
```

## Quick Start

```bash
# Run multi-agent analysis
python main.py --mode multi_agent --data_dir ./realistic_health_data

# Run single LLM comparison
python main.py --mode single_llm --data_dir ./realistic_health_data

# Run ablation studies
python main.py --mode ablation --data_dir ./realistic_health_data

# Run comprehensive comparison
python main.py --mode compare_all --data_dir ./realistic_health_data
```

## Repository Structure

```
health-ai-analysis/
├── main.py                     # Main execution script
├── config/                     # Configuration files
│   ├── __init__.py
│   ├── settings.py            # System configuration
│   └── model_configs.py       # LLM configurations
├── src/                       # Source code
│   ├── __init__.py
│   ├── agents/               # Agent implementations
│   │   ├── __init__.py
│   │   ├── base_agent.py     # Base classes
│   │   ├── health_agents.py  # Multi-agent system
│   │   └── single_llm.py     # Single LLM system
│   ├── data/                 # Data processing
│   │   ├── __init__.py
│   │   ├── processor.py      # Data loading
│   │   └── summarizer.py     # Data summarization
│   ├── evaluation/           # Evaluation framework
│   │   ├── __init__.py
│   │   ├── ablation.py       # Ablation studies
│   │   ├── comparison.py     # Comparisons
│   │   └── metrics.py        # Quality metrics
│   └── utils/               # Utilities
│       ├── __init__.py
│       ├── logger.py        # Logging
│       └── file_manager.py  # File management
├── experiments/             # Experiment runners
│   ├── __init__.py
│   ├── run_ablation.py     # Ablation runner
│   ├── run_comparison.py   # Comparison runner
│   └── analyze_results.py  # Results analysis
├── output/                 # Generated results
├── tests/                  # Test suite
├── requirements.txt        # Dependencies
├── setup.py               # Package setup
└── README.md              # This file
```

## Prerequisites

1. **Python 3.8+**
2. **Ollama** running locally with desired models
3. **Health data files** in Excel format

## Usage Examples

### Multi-Agent Analysis
```python
from src.agents.health_agents import MultiAgentHealthAnalyzer
from config.settings import Config

config = Config(data_dir="./data", model_name="mistral:latest")
analyzer = MultiAgentHealthAnalyzer(config)
analyzer.run_analysis()
```

### Ablation Studies
```python
from experiments.run_ablation import AblationRunner

ablation_runner = AblationRunner(config)
ablation_runner.run_all_ablations()
```

## Configuration

Edit `config/settings.py` to customize:

- Data directories
- Model selection
- Processing options
- Evaluation metrics

## Evaluation Metrics

The system evaluates analysis quality across:

- **Completeness**: Coverage of health domains
- **Clinical Accuracy**: Medical terminology and reasoning
- **Coherence**: Logical flow and consistency
- **Actionability**: Presence of specific recommendations
- **Personalization**: Patient-specific considerations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite: `pytest`
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

```bibtex
@software{health_ai_analysis_2024,
  title={Health AI Analysis System: Multi-Agent Approach to Health Data Interpretation},
  author={Health AI Research Team},
  year={2024}
}
```

# ================================
# INSTALLATION INSTRUCTIONS
# ================================

"""
To download and set up this repository:

1. Create the directory structure:
   mkdir -p health-ai-analysis/{config,src/{agents,data,evaluation,utils},experiments,output,tests}

2. Copy each file section above into the corresponding file path

3. Create __init__.py files:
   touch health-ai-analysis/config/__init__.py
   touch health-ai-analysis/src/__init__.py
   touch health-ai-analysis/src/agents/__init__.py
   touch health-ai-analysis/src/data/__init__.py
   touch health-ai-analysis/src/evaluation/__init__.py
   touch health-ai-analysis/src/utils/__init__.py
   touch health-ai-analysis/experiments/__init__.py
   touch health-ai-analysis/tests/__init__.py

4. Install dependencies:
   cd health-ai-analysis
   pip install -r requirements.txt

5. Run the system:
   python main.py --mode multi_agent --data_dir ./realistic_health_data --verbose

Note: This provides the core framework. Additional implementation files for
complete agents, evaluation metrics, and experiment runners should be added
based on the full artifact code provided earlier.
"""