# ONNX `Normalizer` finite-range stability audit

This repository reproduces ordinary numerical-correctness defects in the
released ONNX Python `ReferenceEvaluator` and ONNX Runtime CPU implementations
of `ai.onnx.ml::Normalizer`.

All counterexamples contain finite values and have ordinary finite normalized
outputs.

## Minimal counterexample

For L2 normalization of two float32 rows:

```text
input                    [[1e30, 1e30], [1e-30, -1e-30]]
exact scaled result      [[0.7071, 0.7071], [0.7071, -0.7071]]
ONNX ReferenceEvaluator [[0, 0], [1, -1]]
ONNX Runtime CPU        [[NaN, NaN], [1e-30, -1e-30]]
```

The exact result is straightforward because normalization is invariant under a
positive common scale. Dividing each row by its largest absolute value before
forming the norm keeps every intermediate bounded.

## Additional failures

- finite float64 magnitudes `1e300` and `1e-300` expose failures in `MAX`, L1
  and L2 paths;
- the released ReferenceEvaluator returns float64 for a double input although
  the operator schema requires a float32 output;
- the released ReferenceEvaluator rejects a valid rank-one integer input by
  attempting to reduce axis 1.

## Cause and corrections

The ONNX reference implementation squares values directly, imposes a fixed
`1e-30` denominator floor, assumes rank two, and does not cast to the schema's
output type. The correction is submitted as
[onnx/onnx#8450](https://github.com/onnx/onnx/pull/8450).

The ONNX Runtime CPU kernel previously formed L2 squares before normalization
and narrowed values or accumulated norms to float. The correction computes
with max-magnitude scaling in double and is submitted as
[microsoft/onnxruntime#32573](https://github.com/microsoft/onnxruntime/pull/32573).

The published operator documentation has a separate mathematical error: its L1
formula omits absolute values, and its L2 formula places `X` inside the square
root. The focused documentation correction is submitted as
[onnx/onnx#8451](https://github.com/onnx/onnx/pull/8451). It deliberately leaves
the separately ambiguous `MAX` semantics unchanged.

## Reproduce

```bash
python3.12 -m venv .venv
.venv/bin/pip install --only-binary=:all: --no-compile -r requirements.txt
.venv/bin/python reproduce.py
```

The script contains assertions for every reported behavior and an independent
scaled oracle.

## Versions and boundary

- reproduced with ONNX `1.22.0`, ONNX Runtime `1.29.0`, NumPy `2.5.2`, Python
  3.12, and the CPU execution provider;
- confirmed against ONNX main commit
  `c9f169adac34bd690bf0d628e9aae7fde3d4be85` and ONNX Runtime main commit
  `a7df32cf6087a11884042a2a95526d72100e3b95` on 2026-09-13;
- mutation tests restore each released algorithm and demonstrate that the new
  regression cases fail;
- this is not a security finding, and no deployed-system impact is claimed.
