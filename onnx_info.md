# how to design your own onnx op?
## 1.some op library 
(1) opset_version
opset_version 表示 ONNX 算子集的版本  
(2) torchscript  
onnx支持torchscipt算子https://pytorch.org/docs/stable/onnx_torchscript_supported_aten_ops.html  
onnx operators(1.16.0): https://onnx.ai/onnx/operators/index.html#
## 2.torch.onnx.export
(1) torch.onnx.export函数
```
torch.onnx.export(
model, 
 
     # 这里的args，是指输入给model的参数，需要传递tuple，因此用括号
     (dummy,), 

     # 储存的文件路径
     "demo.onnx", 

     # 打印详细信息
     verbose=True, 

     # 为输入和输出节点指定名称，方便后面查看或者操作
     input_names=["image"], 
     output_names=["output"], 

     # 这里的opset，指，各类算子以何种方式导出，对应于symbolic_opset11
     opset_version=11, 
 
     # 表示他有batch、height、width3个维度是动态的，在onnx中给其赋值为-1
     dynamic_axes={
         "image": {0: "batch", 2: "height", 3: "width"},
         "output": {0: "batch", 2: "height", 3: "width"},
     }
)
```
Torch.onnx.export执行流程：  
(1.0) trace模式：给定一组输入，再实际执行一遍模型，即把这组输入对应的计算图记录下来，保存为 ONNX 格式
（1.1）如果输入到torch.onnx.export的模型是nn.Module类型，则默认会将模型使用torch.jit.trace转换为ScriptModule

（1.2）使用args参数和torch.jit.trace将模型转换为ScriptModule，torch.jit.trace不能处理模型中的循环和if语句

（1.3）如果模型中存在循环或者if语句，在执行torch.onnx.export之前先使用torch.jit.script将nn.Module转换为ScriptModule  

（2）modify the opset_version and ir_version
```
import onnx
from onnx import version_converter, helper

# import onnxruntime
# help(onnx)

# Preprocessing: load the model to be converted.
model_path = 'old.onnx'
original_model = onnx.load(model_path)
original_model.opset_import[0].version = 11
original_model.ir_version = 6

onnx.save(original_model, "new.onnx")
# print(original_model.Version)


# print('The model before conversion:\n{}'.format(original_model))

```
## 3. how to design your own op in onnx and run it in onnxruntime?
### 3.1 register op in onnx 
```
torch.onnx.register_custom_op_symbolic
```

