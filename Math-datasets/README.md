# Math-datasets

This directory contains loaders for mathematical datasets, particularly focused on math proof problems.

## AoPS-Instruct Dataset

The AoPS-Instruct dataset contains approximately 5k-10k Olympiad-level math proof problems from the Art of Problem Solving (AoPS) forum.

### Dataset Source
- **HuggingFace**: https://huggingface.co/datasets/DeepStudentLlama/AoPS-Instruct
- **GitHub**: https://github.com/DSL-Lab/aops
- **Paper**: https://arxiv.org/abs/2501.14275

### Installation

Make sure you have the `datasets` package installed:

```bash
pip install datasets
```

### Usage

#### Load from HuggingFace (requires internet access)

```python
from Math_datasets import load_aops_dataset, AoPSDataset

# Load 5000 samples from HuggingFace
dataset = load_aops_dataset(sample_size=5000)

# Get problem-solution pairs
pairs = dataset.get_problem_solution_pairs()
print(f"Loaded {len(pairs)} problems")

# Access individual problems
for pair in pairs[:3]:
    print(f"Problem: {pair['problem'][:100]}...")
    print(f"Solution: {pair['solution'][:100]}...")
    print()

# Convert to API format for model input
api_data = dataset.to_api_format()
```

#### Load from local JSONL file

```python
from Math_datasets import AoPSDataset

# Load from a previously saved JSONL file
dataset = AoPSDataset.load_from_jsonl("path/to/aops_data.jsonl")

# Get problem-solution pairs
pairs = dataset.get_problem_solution_pairs()
```

#### Save dataset locally

```python
from Math_datasets import load_aops_dataset

# Load dataset
dataset = load_aops_dataset(sample_size=5000)

# Save raw data
dataset.save_jsonl("data/aops_raw.jsonl")

# Save processed problem-solution pairs
dataset.save_processed_jsonl("data/aops_problems.jsonl")
```

### Command Line Usage

You can also use the module from the command line:

```bash
# Load and save 5000 problems
python Math-datasets/aops_dataset.py --sample-size 5000 --output data/aops_problems.jsonl

# Load with a specific configuration
python Math-datasets/aops_dataset.py --config 2024_not_decontaminated --sample-size 5000

# Just print statistics
python Math-datasets/aops_dataset.py --sample-size 5000 --stats-only

# Use sample data for testing (when HuggingFace is not accessible)
python Math-datasets/aops_dataset.py --use-sample --sample-size 10

# Load from a local file
python Math-datasets/aops_dataset.py --from-jsonl data/aops_data.jsonl
```

### Dataset Configurations

The AoPS-Instruct dataset has two configurations:

1. **default**: Conversational format with `messages` field
   ```json
   {
     "messages": [
       {"role": "user", "content": "Problem text..."},
       {"role": "assistant", "content": "Solution text..."}
     ]
   }
   ```

2. **2024_not_decontaminated**: Format with metadata
   ```json
   {
     "original_question": "...",
     "rewritten_question": "...",
     "original_answers": ["..."],
     "rewritten_answers": ["..."],
     "metadata": {...}
   }
   ```

### API Format

The `to_api_format()` method returns data suitable for LLM API calls:

```python
api_data = dataset.to_api_format()
# Returns:
# [
#   {
#     "messages": [
#       {"role": "user", "content": "Problem text..."},
#       {"role": "assistant", "content": "Solution text..."}
#     ],
#     "id": 0
#   },
#   ...
# ]
```

This format is compatible with OpenAI API and other LLM APIs for fine-tuning or inference.

### References

If you use this dataset, please cite:

```bibtex
@misc{aopsdataset,
      title={Leveraging Online Olympiad-Level Math Problems for LLMs Training and Contamination-Resistant Evaluation}, 
      author={Sadegh Mahdavi and Muchen Li and Kaiwen Liu and Christos Thrampoulidis and Leonid Sigal and Renjie Liao},
      year={2025},
      eprint={2501.14275},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.14275}, 
}
```
