from Py2C import Py2C
import tensorflow as tf
# pyc_lib = Py2C(model_path="Model_WaterMeter_ResNet_251206.h5",
#                torch=False,
#                input_size=(None, 32, 32, 1),
#                type="fxp",
#                fxp_para=(32, 16),
#                num_of_output=1,
#                choose_only_output=True,
#                ide="vs")
# pyc_lib.convert2C()
# pyc_lib.WriteCfile()
# pyc_lib.Write_Float_Weights_File()

model = tf.keras.models.load_model('ResNet-10.h5')

print("Input shape (model.inputs[0].shape):", model.inputs[0].shape)
model.summary()
config = model.get_config()
for i, layer in enumerate(config["layers"]):
    name = layer['config']['name']
    inbound = layer.get('inbound_nodes', [])
    print(f"Layer {i}: {name} - inbound_nodes: {inbound}")