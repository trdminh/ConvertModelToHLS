import tensorflow as tf

old_model = tf.keras.models.load_model('CNN-L06.h5')

new_model = tf.keras.Sequential()

for layer in old_model.layers:
    if not isinstance(layer, tf.keras.layers.Dropout):
        new_model.add(layer)
        
new_model.build(old_model.input_shape)
for i, layer in enumerate(new_model.layers):
    old_layer_idx = 0
    count = 0
    for old_layer in old_model.layers:
        if not isinstance(old_layer, tf.keras.layers.Dropout):
            if count == i:
                old_layer_idx = old_model.layers.index(old_layer)
                break
            count+=1
            
    if len(old_model.layers[old_layer_idx].get_weights()) > 0:
        layer.set_weights(old_model.layers[old_layer_idx].get_weights())
        
new_model.save("CNN-L06-nodropout.h5")