# ML Activity Prediction Analysis

## Overview
Machine learning analysis of EU-OPENSCREEN data:
This project builds machine learning models to predict compound activity (pIC50) from experimental data and molecular descriptors. The goal is not only predict activity, but to understand how model outputs can be used to design more meaningful features.

This project follows an iterative workflow:
- train models on standard descriptors 
- identify important features 
- interpret whether these reflect real chemistry or artefacts 
- use this to guide feature engineering for improved models 


## Project Structure
- Iteration 1: baseline modelling → [link](iteration_1.md)
- Iteration 2: feature engineering → [link](iteration_2.md)

## Key Takeaway
Iteration 1 built the workflow. a compound was removed from the dataset because its very low activity was skewing the model.
