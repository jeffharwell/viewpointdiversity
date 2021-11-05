# Scripts

## Overview

This directory contains scripts that utilize the VDD package to 
do certain tasks. Currently, these tasks are creating 
and saving sets of features and then creating and evaluating 
Support Vector machines that utilize these features.

## Generating and Evaluating Features and Models

One of the significant challenges of this approach, especially 
when trying to use a single computer, is that the vector models
are quite large and when comparing the performance of different 
vector models across different corpora it is easy to run out 
of memory. Additionally, it is nice to be able to separate the
feature generation process, which is quite resource intensive
on its own, with the model training and evaluation, which, 
particularly if you are doing hyperparameter tuning as part of
the evaluation, can take a significant amount of time.

The script 'generate_feature_pickles.py' will parse the 
documents using Spacy, generate the features using the VDD
algorithm and the selected Gensim vector model, and then 
persist the ParsedDocumentsFourForums object and the
FeatureVectorsAndTargets object. The ParsedDocumentsFourForums
contains the entire parsed corpus, so it is quite large. The
FeatureVectorsAndTargets object contains both the 
ParsedDocumentsFourForums object and the vector model, 
so it can be very large. Finally, it creates and saves a
Holder object (viewpointdiversitydetection.FeatureVectorsAndTargets.Holder).
The Holder object only contains the feature vectors and the 
targets, so it is comparatively small. It also contains all
metadata necessary to use the features in a machine 
learning algorithm and output the results in an identifiable
way.

The script 'train_and_evaluate_models.py' does just this task. 
It reads in all the Holder pickles then trains and 
evaluates an SVM using the features in the Holder. It will then
output a Markdown table with the results of the evaluation.
