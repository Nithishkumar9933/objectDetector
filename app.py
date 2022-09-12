import os  #For interacting with the operating system
from flask import Flask, render_template, request, redirect, url_for, send_from_directory #For rendering the template in templates folder
from flask_bootstrap import Bootstrap #To import the bootstrap
from werkzeug import secure_filename #For security purposes
import numpy as np #For numerical calculations
import os
import six.moves.urllib as urllib
import sys
import tensorflow as tf
from collections import defaultdict
from io import StringIO
from PIL import Image #To deal with images
sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as vis_util
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

detection_graph = tf.Graph()