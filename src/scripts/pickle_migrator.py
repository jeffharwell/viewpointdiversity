#!/usr/bin/python

"""
A somewhat silly script. When running scripts inside of the source tree and then saving off object as pickles,
if you use the following import statement:

import src.viewpointdiversitydetection as vdd

instead of

import viewpointdiversitydetection as vdd

Your pickle files will pick up a dependency on the src module and not load successfully outside of this
project. This little script takes vdd.FeatureVectorsAndTargets.Holder objects that have been pickled with a
dependency on the 'src' module and converts them into new pickles that do not have a dependency on the 'src' module.

Hopefully you will never need such a thing. But if you do here is how it is done.
"""

import glob
import sys

# Our SVM and some associated items
import pickle

# General Pieces
import configparser

# Import Viewpoint Diversity Detection Package
#import src.viewpointdiversitydetection as vdd_src
import viewpointdiversitydetection as vdd



#
# Load the feature holder object from a pickle.
#
config = configparser.ConfigParser()
config.read('config.ini')

pickle_directory = config['General']['pickle_directory']

holder_files = glob.glob(f"{pickle_directory}/*holder.pickle")
print(holder_files)
table_strings = []
for filename in holder_files:
    with open(filename, 'rb') as infile:
        oh = pickle.load(infile)
    print(f"Loading feature holder for {oh.topic_name} vector model {oh.vector_model_short_name}")

    new_holder = vdd.Holder(oh.database,
                            oh.topic_name,
                            oh.search_terms,
                            oh.stance_a,
                            oh.stance_b,
                            oh.label_a,
                            oh.label_b,
                            oh.stance_agreement_cutoff, oh.vector_model_short_name)
    new_holder.populate(oh)
    new_filename = f"{pickle_directory}/{oh.topic_name}_{oh.vector_model_short_name}_feature_holder_nosrc.pickle"
    outfile = open(new_filename, 'wb')
    pickle.dump(new_holder, outfile)
    outfile.close()

try:
    print(sys.modules['src'])
    print("The src module was loaded, we loaded pickles that required src as a module")
    print("these pickles would not load successfully outside of the src tree.")
except KeyError as e:
    print("src module was not loaded")