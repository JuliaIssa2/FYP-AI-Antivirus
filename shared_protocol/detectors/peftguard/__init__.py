"""PEFTGuard integration (converter, network, trainer, shared Detector). Needs torch + safetensors + numpy.

Nothing is imported here on purpose, so ``import shared_protocol`` still works without torch. Use the submodules:
``converter``, ``model``, ``train`` (retrain on the shared split), ``detector`` (frozen pure-inference Detector).
"""
