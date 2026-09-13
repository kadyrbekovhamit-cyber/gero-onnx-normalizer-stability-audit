"""Reproduce the three-way semantic conflict for ai.onnx.ml Normalizer MAX."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from reproduce import run_implementation


def main() -> None:
    x = np.array([[-2.0, -1.0], [-3.0, 2.0]], dtype=np.float32)
    reference, runtime = run_implementation(x, "MAX")
    infinity_norm = x / np.max(np.abs(x), axis=1, keepdims=True)
    raw_maximum = x / np.max(x, axis=1, keepdims=True)

    print("input:\n", x)
    print("ONNX ReferenceEvaluator:\n", reference)
    print("ONNX Runtime CPU:\n", runtime)
    print("infinity-norm / scikit-learn semantics:\n", infinity_norm)
    print("raw-maximum / current ONNX prose:\n", raw_maximum)

    assert_allclose(reference, infinity_norm, rtol=0, atol=0)
    assert_allclose(runtime, raw_maximum, rtol=0, atol=0)
    assert not np.allclose(reference, runtime)

    print("\nSemantic conflict reproduced.")


if __name__ == "__main__":
    main()
