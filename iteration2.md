# ml-activity-prediction-analysis

Machine learning analysis of EU-OPENSCREEN data to predict compound activity and identify important molecular features.

See iteration 1 for more background details.

## Objectives

- Reduce the number of features  (screen random removal of features, and focus on the features with the highest importance in the existing model)
- Introduce cross-validation for more robust evaluation  
- Identify important features in the new model, and investigate these to generate more chemically meaningful features (electrostatics, interactions from docking with the target)

Key Findings

1. feature ECP4_1462 is of high importance
2. Compounds containing this structure have structural similarities and are suitable for comparison to identify binding motifs and elucidate some SAR information

Next Steps:
derive more chemically meaningful features from modelling the electrostatics and the interactions of these compounds with the target protein
