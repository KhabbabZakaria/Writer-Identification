#!/usr/bin/env python3
import random

import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

entries = sorted(os.listdir('binarized_trainval'))
#entries = os.listdir('binarized_trainval')
string = 'binarized_trainval'

current_path = os.getcwd()

annotation_range = 100

resize = 1400
centercrop = 800
randomresizedcrop = 400

batch_size = 100
minibatch_size = 10

epochs =100

PATH = "final_model.pt"


#for test dataset
entries_test = sorted(os.listdir('ScriptNet-HistoricalWI-2017-binarized'))
#entries = os.listdir('binarized_test')
string_test = 'ScriptNet-HistoricalWI-2017-binarized'
