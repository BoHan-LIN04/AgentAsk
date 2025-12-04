"""
AoPS-Instruct Dataset Loader

This module provides functionality to load and process the AoPS-Instruct dataset
from HuggingFace (https://huggingface.co/datasets/DeepStudentLlama/AoPS-Instruct).

The AoPS-Instruct dataset contains Olympiad-level math problems collected from
the Art of Problem Solving (AoPS) forum, suitable for training and evaluating
mathematical reasoning in LLMs.

The dataset contains approximately 5k-10k Olympiad-level math proof problems,
making it ideal for training mathematical reasoning capabilities.

Reference:
    - Dataset: https://huggingface.co/datasets/DeepStudentLlama/AoPS-Instruct
    - GitHub: https://github.com/DSL-Lab/aops
    - Paper: https://arxiv.org/abs/2501.14275

Usage:
    # Load from HuggingFace (requires internet access)
    from Math_datasets import load_aops_dataset
    dataset = load_aops_dataset(sample_size=5000)
    
    # Or load from a local JSONL file
    dataset = AoPSDataset.load_from_jsonl("path/to/aops_data.jsonl")
    
    # Get problem-solution pairs for API input
    pairs = dataset.get_problem_solution_pairs()
    api_data = dataset.to_api_format()
"""

import json
import random
from typing import Optional, List, Dict, Any, Union
from pathlib import Path

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    load_dataset = None


class AoPSDataset:
    """
    A dataset loader for the AoPS-Instruct dataset from HuggingFace.
    
    This class provides methods to:
    - Load the dataset from HuggingFace
    - Sample a subset of problems (5k-10k as required)
    - Convert data to a format suitable for API model input
    - Save/load processed data locally
    
    Attributes:
        dataset_name: The HuggingFace dataset identifier
        config_name: The dataset configuration to use ('default' or '2024_not_decontaminated')
        data: The loaded dataset items
    """
    
    DATASET_NAME = "DeepStudentLlama/AoPS-Instruct"
    
    def __init__(
        self,
        config_name: str = "default",
        split: str = "train",
        sample_size: Optional[int] = None,
        seed: int = 42,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the AoPS dataset loader.
        
        Args:
            config_name: Dataset configuration ('default' for conversational format,
                        '2024_not_decontaminated' for format with metadata)
            split: Dataset split to load ('train')
            sample_size: Number of samples to load (None for all, or specify 5000-10000)
            seed: Random seed for sampling
            cache_dir: Optional directory to cache the downloaded dataset
        """
        self.config_name = config_name
        self.split = split
        self.sample_size = sample_size
        self.seed = seed
        self.cache_dir = cache_dir
        self.data: List[Dict[str, Any]] = []
        self._raw_dataset = None
        
        random.seed(seed)
    
    def load(self) -> "AoPSDataset":
        """
        Load the AoPS-Instruct dataset from HuggingFace.
        
        Returns:
            self: The dataset instance with loaded data
            
        Raises:
            ImportError: If the 'datasets' package is not installed
            ConnectionError: If unable to connect to HuggingFace Hub
            RuntimeError: If dataset loading fails
        """
        if not HF_DATASETS_AVAILABLE:
            raise ImportError(
                "The 'datasets' package is required. Install it with: pip install datasets"
            )
        
        print(f"Loading AoPS-Instruct dataset (config={self.config_name}, split={self.split})...")
        print(f"Dataset source: https://huggingface.co/datasets/{self.DATASET_NAME}")
        
        try:
            self._raw_dataset = load_dataset(
                self.DATASET_NAME,
                self.config_name,
                split=self.split,
                cache_dir=self.cache_dir
            )
        except ConnectionError as e:
            print(f"Note: Cannot connect to HuggingFace Hub. You may need to:")
            print(f"  1. Check your internet connection")
            print(f"  2. Set HF_TOKEN environment variable if the dataset requires authentication")
            print(f"  3. Download the dataset manually and use load_from_jsonl() instead")
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset from HuggingFace: {e}")
        
        total_size = len(self._raw_dataset)
        print(f"Dataset loaded: {total_size} total examples")
        
        # Sample if needed
        if self.sample_size and self.sample_size < total_size:
            indices = random.sample(range(total_size), self.sample_size)
            self.data = [self._raw_dataset[i] for i in sorted(indices)]
            print(f"Sampled {self.sample_size} examples")
        else:
            self.data = [self._raw_dataset[i] for i in range(total_size)]
        
        return self
    
    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single example from the dataset.
        
        Args:
            idx: Index of the example
            
        Returns:
            A dictionary containing the example data
        """
        if idx < 0 or idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range [0, {len(self.data)})")
        return self.data[idx]
    
    def get_problem_solution_pairs(self) -> List[Dict[str, str]]:
        """
        Extract problem-solution pairs from the dataset.
        
        This method processes the dataset to extract math problems and their solutions
        in a format suitable for downstream tasks.
        
        Returns:
            A list of dictionaries with 'problem' and 'solution' keys
        """
        pairs = []
        
        for item in self.data:
            # Check if already in problem/solution format (from processed JSONL)
            if "problem" in item and "solution" in item:
                pairs.append({
                    "problem": item["problem"],
                    "solution": item["solution"]
                })
                continue
            
            if self.config_name == "default":
                # Default config has conversational format
                messages = item.get("messages", [])
                if len(messages) >= 2:
                    problem = ""
                    solution = ""
                    for msg in messages:
                        if msg.get("role") == "user":
                            problem = msg.get("content", "")
                        elif msg.get("role") == "assistant":
                            solution = msg.get("content", "")
                    if problem and solution:
                        pairs.append({
                            "problem": problem,
                            "solution": solution
                        })
            else:
                # 2024_not_decontaminated config has metadata format
                problem = item.get("rewritten_question") or item.get("original_question", "")
                solutions = item.get("rewritten_answers") or item.get("original_answers", [])
                if problem and solutions:
                    # Use the first answer as the solution
                    solution = solutions[0] if isinstance(solutions, list) and solutions else str(solutions)
                    pairs.append({
                        "problem": problem,
                        "solution": solution,
                        "metadata": item.get("metadata", {})
                    })
        
        return pairs
    
    def to_api_format(self) -> List[Dict[str, Any]]:
        """
        Convert dataset to a format suitable for API model input.
        
        This method prepares the data for use with LLM APIs by formatting
        problems as user messages and solutions as expected completions.
        
        Returns:
            A list of dictionaries with 'messages' in API-compatible format
        """
        api_data = []
        
        for item in self.data:
            if self.config_name == "default":
                # Already in conversational format
                api_data.append({
                    "messages": item.get("messages", []),
                    "id": len(api_data)
                })
            else:
                # Convert to conversational format
                problem = item.get("rewritten_question") or item.get("original_question", "")
                solutions = item.get("rewritten_answers") or item.get("original_answers", [])
                solution = solutions[0] if isinstance(solutions, list) and solutions else str(solutions)
                
                api_data.append({
                    "messages": [
                        {"role": "user", "content": problem},
                        {"role": "assistant", "content": solution}
                    ],
                    "id": len(api_data),
                    "metadata": item.get("metadata", {})
                })
        
        return api_data
    
    def save_jsonl(self, filepath: Union[str, Path]) -> None:
        """
        Save the dataset to a JSONL file.
        
        Args:
            filepath: Path to save the JSONL file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in self.data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Saved {len(self.data)} examples to {filepath}")
    
    def save_processed_jsonl(self, filepath: Union[str, Path]) -> None:
        """
        Save the processed problem-solution pairs to a JSONL file.
        
        Args:
            filepath: Path to save the JSONL file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        pairs = self.get_problem_solution_pairs()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')
        
        print(f"Saved {len(pairs)} problem-solution pairs to {filepath}")
    
    @classmethod
    def load_from_jsonl(cls, filepath: Union[str, Path]) -> "AoPSDataset":
        """
        Load dataset from a local JSONL file.
        
        Args:
            filepath: Path to the JSONL file
            
        Returns:
            A new AoPSDataset instance with the loaded data
        """
        instance = cls()
        instance.data = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    instance.data.append(json.loads(line))
        
        print(f"Loaded {len(instance.data)} examples from {filepath}")
        return instance
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded dataset.
        
        Returns:
            A dictionary with dataset statistics
        """
        if not self.data:
            return {"total_examples": 0}
        
        pairs = self.get_problem_solution_pairs()
        
        problem_lengths = [len(p["problem"]) for p in pairs if p.get("problem")]
        solution_lengths = [len(p["solution"]) for p in pairs if p.get("solution")]
        
        stats = {
            "total_examples": len(self.data),
            "total_pairs": len(pairs),
            "avg_problem_length": sum(problem_lengths) / len(problem_lengths) if problem_lengths else 0,
            "avg_solution_length": sum(solution_lengths) / len(solution_lengths) if solution_lengths else 0,
            "config_name": self.config_name,
            "sample_size": self.sample_size
        }
        
        return stats

    @classmethod
    def create_sample_data(cls, num_samples: int = 5) -> "AoPSDataset":
        """
        Create a sample dataset with example math problems for testing.
        
        This is useful for testing the data loading pipeline when
        HuggingFace Hub is not accessible.
        
        Args:
            num_samples: Number of sample problems to create
            
        Returns:
            An AoPSDataset instance with sample data
        """
        instance = cls()
        instance.data = []
        
        sample_problems = [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Prove that for any positive integers a and b, gcd(a,b) * lcm(a,b) = a * b."
                    },
                    {
                        "role": "assistant",
                        "content": "Let d = gcd(a,b). We can write a = d*m and b = d*n where gcd(m,n) = 1. "
                                   "Then lcm(a,b) = d*m*n (since m and n are coprime). "
                                   "Therefore, gcd(a,b) * lcm(a,b) = d * (d*m*n) = d²*m*n = (d*m)*(d*n) = a*b. QED."
                    }
                ]
            },
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Prove that √2 is irrational."
                    },
                    {
                        "role": "assistant",
                        "content": "Suppose √2 is rational. Then √2 = p/q where p,q are integers with no common factors. "
                                   "Squaring both sides: 2 = p²/q², so p² = 2q². "
                                   "This means p² is even, so p is even. Let p = 2k. "
                                   "Then 4k² = 2q², so q² = 2k², meaning q is also even. "
                                   "But this contradicts our assumption that p and q have no common factors. "
                                   "Therefore √2 is irrational. QED."
                    }
                ]
            },
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Prove the Pythagorean theorem: In a right triangle, a² + b² = c² where c is the hypotenuse."
                    },
                    {
                        "role": "assistant",
                        "content": "Consider a square with side length (a+b). Inside, place four copies of our right triangle "
                                   "with legs a and b. The area of the large square is (a+b)². "
                                   "The four triangles have total area 4*(1/2)*a*b = 2ab. "
                                   "The remaining area in the center is a square with side c, so its area is c². "
                                   "Therefore: (a+b)² = 2ab + c². Expanding: a² + 2ab + b² = 2ab + c². "
                                   "Simplifying: a² + b² = c². QED."
                    }
                ]
            },
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Prove that the sum of the first n positive integers is n(n+1)/2."
                    },
                    {
                        "role": "assistant",
                        "content": "We prove this by induction. Base case: n=1, sum = 1 = 1*2/2. ✓ "
                                   "Inductive step: Assume true for n=k, i.e., 1+2+...+k = k(k+1)/2. "
                                   "For n=k+1: 1+2+...+k+(k+1) = k(k+1)/2 + (k+1) = (k+1)(k/2 + 1) = (k+1)(k+2)/2. "
                                   "This matches the formula for n=k+1. By induction, the formula holds for all n. QED."
                    }
                ]
            },
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Prove that there are infinitely many prime numbers."
                    },
                    {
                        "role": "assistant",
                        "content": "Suppose there are finitely many primes: p₁, p₂, ..., pₙ. "
                                   "Consider N = p₁*p₂*...*pₙ + 1. "
                                   "N is not divisible by any pᵢ (remainder is always 1). "
                                   "So either N is prime, or N has a prime factor not in our list. "
                                   "Either way, we have a prime not in our finite list. "
                                   "This contradicts our assumption. Therefore there are infinitely many primes. QED."
                    }
                ]
            }
        ]
        
        # Create the requested number of samples
        for i in range(num_samples):
            instance.data.append(sample_problems[i % len(sample_problems)])
        
        print(f"Created {len(instance.data)} sample examples")
        return instance


def load_aops_dataset(
    sample_size: Optional[int] = None,
    config_name: str = "default",
    split: str = "train",
    seed: int = 42,
    cache_dir: Optional[str] = None
) -> AoPSDataset:
    """
    Convenience function to load the AoPS-Instruct dataset.
    
    Args:
        sample_size: Number of samples to load (None for all, or 5000-10000 recommended)
        config_name: Dataset configuration ('default' or '2024_not_decontaminated')
        split: Dataset split to load
        seed: Random seed for sampling
        cache_dir: Optional directory to cache the downloaded dataset
        
    Returns:
        An AoPSDataset instance with the loaded data
        
    Example:
        >>> dataset = load_aops_dataset(sample_size=5000)
        >>> print(f"Loaded {len(dataset)} examples")
        >>> pairs = dataset.get_problem_solution_pairs()
        >>> print(f"First problem: {pairs[0]['problem'][:100]}...")
    """
    dataset = AoPSDataset(
        config_name=config_name,
        split=split,
        sample_size=sample_size,
        seed=seed,
        cache_dir=cache_dir
    )
    return dataset.load()


# Main entry point for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load and process AoPS-Instruct dataset")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Number of samples to load (default: 5000, use 0 for all)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        choices=["default", "2024_not_decontaminated"],
        help="Dataset configuration to use"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="aops_dataset.jsonl",
        help="Output file path for saving processed data"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only print statistics without saving"
    )
    parser.add_argument(
        "--use-sample",
        action="store_true",
        help="Use sample data instead of loading from HuggingFace (for testing)"
    )
    parser.add_argument(
        "--from-jsonl",
        type=str,
        default=None,
        help="Load from a local JSONL file instead of HuggingFace"
    )
    
    args = parser.parse_args()
    
    # Load dataset based on options
    sample_size = args.sample_size if args.sample_size > 0 else None
    
    if args.from_jsonl:
        print(f"Loading from local file: {args.from_jsonl}")
        dataset = AoPSDataset.load_from_jsonl(args.from_jsonl)
    elif args.use_sample:
        print("Using sample data for testing...")
        dataset = AoPSDataset.create_sample_data(num_samples=sample_size or 5)
    else:
        try:
            dataset = load_aops_dataset(
                sample_size=sample_size,
                config_name=args.config
            )
        except (ConnectionError, RuntimeError) as e:
            print(f"\nWarning: Could not load from HuggingFace: {e}")
            print("Falling back to sample data for demonstration...")
            dataset = AoPSDataset.create_sample_data(num_samples=5)
    
    # Print statistics
    stats = dataset.get_statistics()
    print("\n=== Dataset Statistics ===")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Save if not stats-only
    if not args.stats_only:
        dataset.save_processed_jsonl(args.output)
        print(f"\nDataset saved to {args.output}")
    
    # Show sample
    pairs = dataset.get_problem_solution_pairs()
    if pairs:
        print("\n=== Sample Problem ===")
        print(f"Problem: {pairs[0]['problem'][:500]}...")
        print(f"\nSolution: {pairs[0]['solution'][:500]}...")
