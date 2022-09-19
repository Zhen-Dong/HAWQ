# Export HAWQ to QONNX

### Export Model
```python 
from utils.export import ExportManager

...

manager = ExportManager(hawq_model)
manager.export(
    torch.randn([1, 16]),  # input for tracing 
    "hawq2qonnx_model.onnx"
)
```


### Execute ONNX graph with QONNX operators
```python
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx

...

qonnx_model = ModelWrapper("hawq2qonnx_model.onnx")

input_dict = {"global_in": X_test}  

output_dict = execute_onnx(qonnx_model, input_dict)
```