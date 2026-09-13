# `ai.onnx.ml::Normalizer` MAX has conflicting semantics across the spec, ReferenceEvaluator, and ONNX Runtime

## Is the issue related to model conversion?

Yes, but the conflict is in the ONNX operator contract itself. `sklearn-onnx`
maps `sklearn.preprocessing.Normalizer(norm="max")` directly to
`ai.onnx.ml::Normalizer(norm="MAX")`.

## Describe the bug

`Normalizer` currently has two incompatible meanings for `MAX`:

1. The ONNX schema says `Y = X / max(X)`.
2. ONNX ReferenceEvaluator divides by `max(abs(X))`.
3. ONNX Runtime CPU divides by `max(X)`.
4. scikit-learn defines its `max` normalizer as division by
   `max(abs(X))`, and `sklearn-onnx` maps that mode directly to ONNX `MAX`.

The difference is observable for any row whose largest-magnitude value is
negative. Conversion parity and backend conformance cannot both hold under the
current contract.

At the current pinned revisions:

- [ONNX schema](https://github.com/onnx/onnx/blob/c9f169adac34bd690bf0d628e9aae7fde3d4be85/onnx/defs/doc_strings.cc#L6003-L6009):
  `Max: Y = X / max(X)`.
- [ONNX ReferenceEvaluator](https://github.com/onnx/onnx/blob/c9f169adac34bd690bf0d628e9aae7fde3d4be85/onnx/reference/ops/aionnxml/op_normalizer.py#L11-L16):
  `np.abs(x).max(axis=1)`.
- [ONNX Runtime CPU](https://github.com/microsoft/onnxruntime/blob/a7df32cf6087a11884042a2a95526d72100e3b95/onnxruntime/core/providers/cpu/ml/normalizer.cc#L45-L62):
  `std::max(max, value)`.
- [scikit-learn](https://github.com/scikit-learn/scikit-learn/blob/dd3ca57300e14d45b7a34fccd0165d143c7a364c/sklearn/preprocessing/_data.py#L2074-L2082):
  `xp.max(xp.abs(X), axis=1)`.
- [sklearn-onnx](https://github.com/onnx/sklearn-onnx/blob/449ec0526152514345a00370356131e325b9abc0/skl2onnx/operator_converters/normaliser.py#L22-L39):
  `{"max": "MAX", "l1": "L1", "l2": "L2"}`.

## System information

- macOS arm64
- Python 3.12
- onnx 1.22.0
- onnxruntime 1.29.0 (CPUExecutionProvider)
- numpy 2.5.2

## Reproduction instructions

```python
import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator

x = np.array([[-2.0, -1.0], [-3.0, 2.0]], dtype=np.float32)
node = helper.make_node(
    "Normalizer", ["X"], ["Y"], domain="ai.onnx.ml", norm="MAX"
)
graph = helper.make_graph(
    [node],
    "normalizer_max_negative",
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2])],
)
model = helper.make_model(
    graph,
    opset_imports=[helper.make_opsetid("", 22), helper.make_opsetid("ai.onnx.ml", 5)],
    ir_version=10,
)

print(ReferenceEvaluator(model).run(None, {"X": x})[0])
print(ort.InferenceSession(model.SerializeToString()).run(None, {"X": x})[0])
```

Released output:

```text
ReferenceEvaluator:
[[-1.        -0.5      ]
 [-1.         0.6666667]]

ONNX Runtime CPU:
[[ 2.   1. ]
 [-1.5  1. ]]
```

An independent reproducer with assertions is available at:
https://github.com/kadyrbekovhamit-cyber/gero-onnx-normalizer-stability-audit/blob/main/reproduce_max_semantic_conflict.py

## Expected behavior

The specification, ReferenceEvaluator, ONNX Runtime, and standard converter
should agree.

I recommend defining `MAX` as the infinity norm,
`Y = X / max(abs(X))`, because:

- that is a norm and preserves signs;
- it matches scikit-learn, the source framework used by the standard converter;
- it matches the existing ONNX ReferenceEvaluator behavior;
- the current raw-maximum behavior can produce values with magnitude greater
  than one and flips an all-negative row to positive values.

However, changing the normative meaning may affect existing models that rely on
ONNX Runtime's current behavior. I am filing this as a specification
clarification before proposing a behavioral patch. Once the intended contract
is confirmed, I am willing to contribute the schema, conformance tests,
ReferenceEvaluator, converter, and/or ONNX Runtime changes needed to make the
stack consistent.

## Notes

I searched open and closed issues and pull requests in ONNX, ONNX Runtime, and
sklearn-onnx using combinations of `Normalizer`, `MAX`, `negative`, and
`maximum absolute`, and did not find a duplicate.
