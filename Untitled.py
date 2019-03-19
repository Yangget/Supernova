#!/usr/bin/env python
# coding: utf-8

# In[1]:


##keras_retinanet/bin/convert_model.py snapshots/resnet50_csv_01.h5 model/model.h5


# In[2]:


from keras_retinanet.models import load_model
model = load_model('model/model_5.h5', backbone_name='resnet50')


# In[3]:


# show images inline
get_ipython().run_line_magic('matplotlib', 'inline')

# automatically reload modules when they have changed
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf


# In[4]:


image = read_image_bgr('data_new/train/ed155979ee6f796884898fc40230bf2b_c.jpg')


# In[5]:


# copy to draw on
draw = image.copy()
draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

# preprocess image for network
image = preprocess_image(image)
image, scale = resize_image(image)


# In[6]:


# process image
start = time.time()
boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
print("processing time: ", time.time() - start)


# In[7]:


# correct for image scale
boxes /= scale
# boxes
boxes[0]
scores[0]
labels[0]


# In[8]:


labels_to_names={0:"newtarget",
                 1:"isstar",
                 2:"ghost",
                 3:"known",
                 4:"pity",
                 5:"noise",
                 6:"asteroid",
                 7:"isnova",
                }

for box, score, label in zip(boxes[0], scores[0], labels[0]):
    if label  == 1 or label == 0 or label == 3 or label == 6 or label == 7:
        print(label,box)


# In[9]:


# visualize detections
i = 0
for box, score, label in zip(boxes[0], scores[0], labels[0]):
    # scores are sorted so we can break
    
# #     if score < 0.5:
# #         break
#     if i > 3:    
#         break
    if label  == 1 or label == 0 or label == 3 or label == 6 or label == 7:
        i += 1
        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)
        
        print(box)
        if i == 5:
            break
        
plt.figure(figsize=(15, 15))
plt.axis('off')
plt.imshow(draw)
plt.show()


# In[ ]:





# In[ ]:




