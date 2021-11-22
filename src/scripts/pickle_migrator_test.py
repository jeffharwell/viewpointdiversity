#!/usr/bin/python

"""
Loads a set of pickles and then checks to see if the 'src' module was loaded. If it was then the pickles
depend on the 'src' module and will only work within this project. If not then the pickles do not depend
on the 'src' module and so they will work outside of this project as long as the viewpointdiversitydetection module
is available.
"""

import glob
import sys

# Our SVM and some associated items
import pickle

# General Pieces
import configparser

# Import Viewpoint Diversity Detection Package
import viewpointdiversitydetection as vdd

#
# Load the feature holder object from a pickle.
#
config = configparser.ConfigParser()
config.read('config.ini')

pickle_directory = config['General']['pickle_directory']

holder_files = glob.glob(f"{pickle_directory}/*feature_holder_nosrc.pickle")
print(holder_files)
table_strings = []
for filename in holder_files:
    with open(filename, 'rb') as infile:
        oh = pickle.load(infile)
    print(f"Loading features for {oh.topic_name} vector model {oh.vector_model_short_name}")

try:
    print(sys.modules['src'])
except KeyError as e:
    print("src module was not loaded. Test passed, this pickle will work with the vdd module outside the src tree.")

