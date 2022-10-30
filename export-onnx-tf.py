import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load("weights/PlateNumberDetect_yolov7_best.onnx")


tf_rep = prepare(onnx_model)

tf_rep.export_graph("weights/PlateNumberDetect_yolov7_best.tf")