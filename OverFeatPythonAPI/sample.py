import overfeatfunctions
import numpy as np

image = np.zeros((3, 231, 231), dtype=np.float32)
output = overfeatfunctions.first_layer(image,
        'OverFeat/data/default/net_weight_0', 0)
print output.shape
