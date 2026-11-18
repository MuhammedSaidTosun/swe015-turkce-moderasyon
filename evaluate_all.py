"""Measure the complete five-model comparison on the official held-out split."""

from evaluate_neural import main


if __name__ == "__main__":
    main(include_transformers=True)
