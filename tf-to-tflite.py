import tensorflow as tf
# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model("weights")
tflite_model = converter.convert()

# Save the model
with open("weights/yolov7_tiny_plate_best.tflite", 'wb') as f:
    f.write(tflite_model)