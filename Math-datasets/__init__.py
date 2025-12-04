"""
Math-datasets module for loading and processing mathematical datasets.

This module provides loaders for various math proof and problem-solving datasets,
starting with the AoPS-Instruct dataset from HuggingFace.

Available datasets:
    - AoPS-Instruct: Olympiad-level math problems from Art of Problem Solving forum

Usage:
    from Math_datasets import load_aops_dataset, AoPSDataset
    
    # Load a sample of 5000 problems
    dataset = load_aops_dataset(sample_size=5000)
    
    # Get problem-solution pairs
    pairs = dataset.get_problem_solution_pairs()
    
    # Convert to API format for model input
    api_data = dataset.to_api_format()
"""

from .aops_dataset import AoPSDataset, load_aops_dataset

__all__ = [
    "AoPSDataset",
    "load_aops_dataset",
]
