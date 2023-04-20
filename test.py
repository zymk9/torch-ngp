import numpy as np
import cv2
f_path = '/disk1/yliugu/Grounded-Segment-Anything/data/nerf_llff_data/fern/features/image000.npz'
with np.load(f_path) as data:
    res = data['res']
    feature = data['embedding']
    feature = feature.reshape(res.tolist())
t = (300, 1000)
# res = cv2.resize(feature.transpose(1,2,0), dsize=t, interpolation=cv2.INTER_CUBIC)
print(feature[:, 0,0].std() )
