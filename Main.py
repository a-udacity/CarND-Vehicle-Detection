# In[1]:

### Main project file to Classify, Extract features, Create Model and Process Video


# In[2]:

# get_ipython().magic('matplotlib inline')
# get_ipython().magic('reload_ext autoreload')
# get_ipython().magic('autoreload 2')


# In[3]:

from process import Process
from vehicleclassifier import VehicleClassifier

# ### Features

# In[6]:

clf = VehicleClassifier()
clf.create_features()
clf.load_features()


# ### Model

# In[7]:

clf.create_model()


# ### Process test images

# In[ ]:

process = Process()
processor_function = process.process_image


# ### Process video

# In[ ]:

process.process_video()
