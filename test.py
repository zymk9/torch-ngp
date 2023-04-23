import numpy as np
import cv2
f_path = '/disk1/yliugu/torch-ngp/workspace/fern_sam_crop/validation/ngp_ep0080_0001_feature.npz'
with np.load(f_path) as data:
    res = data['res']
    feature = data['embedding']
    feature = feature.reshape(res.tolist())
t = (300, 1000)
# res = cv2.resize(feature.transpose(1,2,0), dsize=t, interpolation=cv2.INTER_CUBIC)
print(feature )
