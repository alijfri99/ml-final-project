import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf

from Utilities import FilesHandling

train_ds = FilesHandling.read_images("train")

class_names = train_ds.class_names
print(class_names)


