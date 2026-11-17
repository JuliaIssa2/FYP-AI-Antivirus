"""Thin shared Detector adapter for the named saved-feature spectral baseline."""

from dataclasses import dataclass, field

from ...manifest import ProtocolError
from ...scoring import AdapterInput, DetectorSpec, ProtocolBinding, finite_score
from .features import FeatureSet
from .model import FrozenSpectralModel


@dataclass(frozen=True)
class SpectralDetector:
    model: FrozenSpectralModel
    binding: ProtocolBinding
    features: FeatureSet
    backend: str = "sklearn"
    _spec: DetectorSpec = field(init=False, repr=False)

    def __post_init__(self):
        self.model.verify(self.binding)
        if (self.features.protocol_sha256 != self.binding.sha256
                or self.features.records_sha256 != self.binding.records_digest(self.features.partition)
                or tuple(row.adapter_id for row in self.features.records) != self.binding.ids(
                    self.features.partition, representatives=self.features.partition == "calibration")):
            raise ProtocolError("Feature view does not match this frozen protocol")
        object.__setattr__(self, "_spec", self.model.spec(self.binding, self.backend))

    @property
    def spec(self):
        return self._spec

    def score(self, adapters: tuple[AdapterInput, ...]) -> dict[str, float]:
        if type(adapters) is not tuple or adapters != self.features.allowed_inputs:
            raise ProtocolError("Score exactly the authorized stage view through shared scoring APIs")
        vectors = [self.features.get(adapter) for adapter in adapters]
        scores = self.model.predict(vectors, backend=self.backend)
        if len(scores) != len(adapters):
            raise ProtocolError("Model returned the wrong number of probabilities")
        result = {}
        for adapter, value in zip(adapters, scores):
            value = finite_score(value)
            if not 0 <= value <= 1:
                raise ProtocolError("Baseline must return the probability of malicious class 1")
            result[adapter.adapter_id] = value
        return result
