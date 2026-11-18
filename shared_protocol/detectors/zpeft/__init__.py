"""Z-PEFT-style detector (head-wise spectral features + logistic regression). Needs numpy, safetensors, scikit-learn.

Nothing is imported here on purpose. Submodules: ``features``, ``model`` (frozen JSON state), ``train`` (fit on the shared
train split), ``detector`` (frozen pure-inference Detector).
"""
