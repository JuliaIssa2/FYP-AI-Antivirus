"""Benign tasks: dataset, split, and conversion to alpaca-style {instruction, input, output} rows.

Kept free of torch/transformers so the exporter and row ledger can import it on a laptop.
"""
AG_NEWS = ["World", "Sports", "Business", "Sci/Tech"]

DATASETS = {  # task -> (Hugging Face dataset id, split)
    "alpaca": ("yahma/alpaca-cleaned", "train"),
    "code": ("sahil2801/CodeAlpaca-20k", "train"),
    "dolly": ("databricks/databricks-dolly-15k", "train"),
    "ag_news": ("fancyzhx/ag_news", "train"),
    "sst2": ("stanfordnlp/sst2", "train"),
}

CONVERT = {
    "alpaca": lambda r: {"instruction": r["instruction"], "input": r.get("input", ""), "output": r["output"]},
    "code": lambda r: {"instruction": r["instruction"], "input": r.get("input", ""), "output": r["output"]},
    "dolly": lambda r: {"instruction": r["instruction"], "input": r.get("context", ""), "output": r["response"]},
    "ag_news": lambda r: {"instruction": "Classify the topic of this news article. Answer with one of: "
                                         "World, Sports, Business, Sci/Tech.",
                          "input": r["text"], "output": AG_NEWS[r["label"]]},
    "sst2": lambda r: {"instruction": "Classify the sentiment of this sentence. Answer with positive or negative.",
                       "input": r["sentence"], "output": ["negative", "positive"][r["label"]]},
}


def seed_rows(num_rows, seed, count):
    """Row indices train_adapter uses for a seed: datasets' shuffle(seed) is
    np.random.default_rng(seed).permutation(len(ds)); the first `count` rows are taken."""
    import numpy as np
    return [int(i) for i in np.random.default_rng(seed).permutation(num_rows)[:count]]
