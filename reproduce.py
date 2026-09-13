"""Reproduce finite-range failures in released ONNX Normalizer implementations."""

from __future__ import annotations

import warnings

import numpy as np
import onnxruntime as ort
from numpy.exceptions import AxisError
from numpy.testing import assert_allclose
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


DTYPE_TO_TENSOR = {
    np.dtype(np.float32): TensorProto.FLOAT,
    np.dtype(np.float64): TensorProto.DOUBLE,
    np.dtype(np.int32): TensorProto.INT32,
    np.dtype(np.int64): TensorProto.INT64,
}


def make_model(x: np.ndarray, norm: str):
    tensor_type = DTYPE_TO_TENSOR[x.dtype]
    node = helper.make_node(
        "Normalizer", ["X"], ["Y"], domain="ai.onnx.ml", norm=norm
    )
    graph = helper.make_graph(
        [node],
        f"normalizer_{norm.lower()}_{x.dtype}",
        [helper.make_tensor_value_info("X", tensor_type, list(x.shape))],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, list(x.shape))],
    )
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 22),
            helper.make_opsetid("ai.onnx.ml", 5),
        ],
        ir_version=10,
    )


def scaled_oracle(x: np.ndarray, norm: str) -> np.ndarray:
    """Independent finite-input oracle with bounded intermediate values."""
    work = x.astype(np.float64)
    axis = 0 if x.ndim == 1 else 1
    scale = np.max(np.abs(work), axis=axis, keepdims=True)
    scaled = np.divide(work, scale, out=work.copy(), where=scale != 0)

    if norm == "MAX":
        result = scaled
    elif norm == "L1":
        divisor = np.sum(np.abs(scaled), axis=axis, keepdims=True)
        result = np.divide(scaled, divisor, out=scaled.copy(), where=divisor != 0)
    elif norm == "L2":
        divisor = np.sqrt(np.sum(scaled * scaled, axis=axis, keepdims=True))
        result = np.divide(scaled, divisor, out=scaled.copy(), where=divisor != 0)
    else:
        raise ValueError(norm)

    return result.astype(np.float32)


def run_implementation(x: np.ndarray, norm: str):
    model = make_model(x, norm)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        reference = ReferenceEvaluator(model).run(None, {"X": x})[0]
        runtime = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        ).run(None, {"X": x})[0]
    return reference, runtime


def main() -> None:
    float_extremes = np.array(
        [[1e30, 1e30], [1e-30, -1e-30]], dtype=np.float32
    )
    reference, runtime = run_implementation(float_extremes, "L2")
    exact = scaled_oracle(float_extremes, "L2")

    print("FLOAT32 L2")
    print("input:\n", float_extremes)
    print("ONNX ReferenceEvaluator:\n", reference)
    print("ONNX Runtime CPU:\n", runtime)
    print("scaled oracle:\n", exact)

    assert_allclose(exact, [[2**-0.5, 2**-0.5], [2**-0.5, -(2**-0.5)]])
    assert not np.all(np.isfinite(runtime[0]))
    assert_allclose(runtime[1], float_extremes[1], rtol=0, atol=0)
    assert_allclose(reference[0], [0.0, 0.0], rtol=0, atol=0)
    assert_allclose(reference[1], [1.0, -1.0], rtol=0, atol=0)

    double_extremes = np.array(
        [[1e300, 1e300], [1e-300, -1e-300]], dtype=np.float64
    )
    print("\nFLOAT64 FINITE RANGE")
    for norm in ("MAX", "L1", "L2"):
        reference, runtime = run_implementation(double_extremes, norm)
        exact = scaled_oracle(double_extremes, norm)
        print(f"{norm} reference dtype={reference.dtype}:\n", reference)
        print(f"{norm} runtime dtype={runtime.dtype}:\n", runtime)
        print(f"{norm} oracle dtype={exact.dtype}:\n", exact)

        assert reference.dtype == np.float64  # schema requires tensor(float)
        assert runtime.dtype == np.float32
        assert exact.dtype == np.float32
        assert not np.allclose(reference, exact, equal_nan=True)
        assert not np.allclose(runtime, exact, equal_nan=True)

    one_dimensional = np.array([3, 4], dtype=np.int32)
    model = make_model(one_dimensional, "L2")
    try:
        ReferenceEvaluator(model).run(None, {"X": one_dimensional})
    except AxisError as exc:
        print("\nINT32 RANK-1 ReferenceEvaluator error:", exc)
    else:
        raise AssertionError("released ReferenceEvaluator unexpectedly accepted rank-1 input")

    print("\nAll released-behavior checks reproduced.")


if __name__ == "__main__":
    main()
