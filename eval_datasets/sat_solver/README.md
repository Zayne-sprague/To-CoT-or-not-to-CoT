## How to Add a Dataset for SatLM

1. Implement a symbolic version of the original CoT dataset, see `GSM8KSymDataset` in `eval_datasets/types/gsm8k.py` for an example.
2. Update the files for importing and dataset types according. Spefically, update `eval_datasets/__init__.py`, `eval_datasets/reasoning_dataset.py`, and `experiments/utils.py`.
3. Implement the SAT solving procedure.