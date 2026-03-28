Terms:
- Strategy: weight by score / uniform weight
- Environment strategy:
  + same environment (E1): using all the query images from all environments of test sample but only query with the same environment as the database sample will be used for prediction
  + all environment (E2): using all the query images from all environments of test sample and all of them will be used for prediction regardless of the environment of the database sample
  + E3_{environment name}: using only the query images from the same environment as the database sample for prediction
  + E4_{environment name}: exclude the query images from the environment as the database sample for prediction

Phase 1: Setup and dataset preparation [DONE]
1. Read and analyze current project, double check all the scripts then update #file:README.md for anyone can be able to reproduce the project and understand the project structure and how to use it.
2. Dataset downloaded and reformatted into `Dataset/hierarchical/`.

Phase 2: Weights and evaluation
1. Generate fold mapping files from `Dataset/strain_to_specy.csv` into 5 files: `Dataset/strain_to_specy_fold0.csv` to `Dataset/strain_to_specy_fold4.csv`.
2. For each fold, retrain EfficientNetB1 finetuned with that fold mapping split (train on `Test=False`, validate on `Test=True`) and save weights as `fold{idx}_EfficientNetB1_finetuned.pth`.
3. For each fold training run, export progress and validation artifacts in the same weights folder:
  - history JSON
  - training/validation accuracy-loss curve image
  - confusion matrix on fold validation test strains
  - per-class accuracy chart
4. Start new qdrant using #file:./docker-compose.yml and use #file:src/scripts/upload_finetuned_features.py after generating fold-specific json features.
5. Test evaluation script for one fold with only EfficientNetB1 finetuned, one aggregation strategy, and one environment strategy to verify end-to-end fold flow.

Phase 3: Cross validation and benchmarking
We use cross validation in strain-level which means we will rotate the testset. There will be one strain for each species is selected to be in the testset and each species now contains 4 to 7 strains (mostly 4). So we will have 5 fold (1 fold will have the same of some strains for only 4 strain species).
- Model extractor will be fixed to efficientb1 finetuned.
- Environment strategy will be assessed between all environment (E2) and same environment (E1).
- Strategy will be both (weight by score and uniform weight).
- K will be 3, 5, 7, 9, 11, 13, 15 to see the effect of K on the performance. Why we should not choose a very large K? We have very few data samples.
- We will brute force all the combinations of the above settings and report the performance in each setting. We will also compare the performance between different settings to see which one is better.
- The results will be just like the output of our #file:src/main.py evaluation command.

1. Write a script to automate the process and track the progress. 
2. After the script is done, have a summary table in csv append every batch of the results.
3. Make sure each fold uses its own finetuned weight and its own Qdrant fold collection (`*_fold{idx}`), then summarize and visualize all results in a structured way.
4. Visualize export to #file:report/week_1_2/ images folder with appropriate naming.

Phase 4: Write report for our work
1. Write report following #file:report_dod.md Definition of Done.
2. Overview section must include:
  - quick intro of problem
  - glossary for terms
  - purpose of cross validation to choose best parameters
3. Methodology section must include:
  - number of folds and fold split explanation
  - why strain-level split is required
  - visualization of 5 folds
  - clear method diagram
4. Implementation section must include:
  - vector database retrieval/prediction pipeline brief with diagram
  - cross_validation script block diagram (Mermaid) and function-level explanation
5. Results section must include:
  - comparison charts for all hyperparameter configurations with captions
  - best configuration selection and analysis of performance differences
6. Conclusion section must summarize key outcomes and practical recommendation.