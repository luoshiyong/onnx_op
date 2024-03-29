import onnx
import torch 
import os
import sys
# 把该路径加入环境变量/home/luoshiyong/work_dirs/algo/onnx_op
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from net.net1 import Model
# 这个包对应opset11的导出代码，如果想修改导出的细节，可以在这里修改代码
# import torch.onnx.symbolic_opset11
print("对应opset文件夹代码在这里：", os.path.dirname(torch.onnx.__file__))
model = Model()
dummy = torch.zeros(1, 1, 3, 3)

torch.onnx.export(
     model, 
 
     # 这里的args，是指输入给model的参数，需要传递tuple，因此用括号
     (dummy,), 

     # 储存的文件路径
     "../onnx/demo.onnx", 

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
def onnx_check(model_path):
    """
    验证导出的模型格式时候正确
    :param model_path:
    :return:
    """
    print("onnx model checking------------------------>")
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    print("show-------->onnx_model.graph\n",onnx.helper.printable_graph(onnx_model.graph))
onnx_check("demo.onnx")
print("Done.!")
