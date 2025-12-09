= Evaluation

== Experimental Setup

The evaluation was conducted using a comprehensive testing framework with the following configuration:

- *K-Nearest Neighbors (K):* 7
- *Minimum Samples:* None (all species considere=== Case 1: Best Overall Performance

#figure(
  image("assets/results/comprehensive_k7_NoSib/ColorHistogramHS_E2_AVG/confusion_matrix_20251113_153320.png", width: 70%),
  caption: [ColorHistogramHS with E2 strategy and AVG aggregation achieved the highest accuracy at 75% (36/48 correct predictions).]
)

ColorHistogramHS E2 AVG demonstrated superior performance by effectively leveraging color information in the Hue-Saturation space across different growth environments. The cross-environment training (E2) provided robust features that generalized well, correctly identifying 36 out of 48 test cases. The confusion matrix shows strong diagonal values indicating reliable species identification, though some confusion persists between morphologically similar species like P. neoechinulatum and P. freii.

#figure(
  grid(
    columns: 2,
    gutter: 10pt,
    image("assets/results/comprehensive_k7_NoSib/ColorHistogramHS_E2_AVG/false_predictions/DTO_148-C8_3efbd33ab2e8_pred_Penicillium_neoechin.jpg", width: 100%),
    image("assets/results/comprehensive_k7_NoSib/ColorHistogramHS_E2_AVG/false_predictions/DTO_158-D1_5966b68c1ce8_pred_Penicillium_freii.jpg", width: 100%),
  ),
  caption: [Example false predictions: (Left) DTO-148-C8 misclassified as P. neoechinulatum, (Right) DTO-158-D1 misclassified as P. freii.]
)bling Filtering:* Enabled (excluding images from the same parent)
- *Test Strains:* 8 species, 6 test sets per strain (48 predictions per configuration)

=== Evaluation Strategies

The evaluation employed three distinct environment strategies to assess model performance under different conditions:

*E1 (Same Environment Query):* For each strain, test sets were created with one image from each of the 7 growth environments (CREA, CYA, CYA30, CYAS, DG18, MEA, YES). Each query image searches for neighbors only within its own environment. This strategy evaluates the model's ability to identify species when both training and test data are from the same growth conditions. Each test set contains 7 images (one per environment), and 6 test sets are generated per strain, resulting in 48 predictions per configuration.

*E2 (Cross-Environment Query):* Similar to E1, test sets contain one image from each environment. However, query images search across all environments without filtering. This strategy assesses the model's robustness to environmental variations and tests whether features learned from one environment can generalize to identify species grown in different conditions. Like E1, this produces 48 predictions per configuration.

*E3 (Single Environment Query):* Test sets contain images from only one specific environment (e.g., E3_CYA tests only on CYA environment). Each test set has one image, and 6 test sets are generated per strain. Queries search within the same environment. This strategy evaluates environment-specific performance and identifies which growth conditions provide the most discriminative features for species identification. Seven separate E3 evaluations were conducted (one per environment), each producing 48 predictions.

=== Feature Extractors

Seven feature extraction methods were evaluated, representing different approaches to image analysis:

*Traditional Computer Vision Methods:*
- *ColorHistogram:* RGB color distribution analysis capturing color patterns in 256-bin histograms
- *ColorHistogramHS:* Hue-Saturation color space histograms, more invariant to lighting changes
- *HOG (Histogram of Oriented Gradients):* Edge and texture patterns through gradient orientation analysis
- *Gabor Filters:* Multi-scale texture analysis using oriented Gabor wavelets

*Deep Learning Methods:*
- *ResNet50:* 50-layer residual network pre-trained on ImageNet, 2048-dimensional features
- *MobileNetV2:* Lightweight CNN optimized for efficiency, 1280-dimensional features
- *EfficientNetV2B0:* Compound-scaled efficient architecture, 1280-dimensional features

=== Aggregation Strategies

Two voting aggregation strategies were tested for combining predictions from multiple query images:

*S1 (AVG - Weighted Voting):* Each neighbor's vote is weighted by its similarity score (cosine similarity). Species with higher cumulative weighted scores are ranked higher. This approach gives more influence to highly similar neighbors.

*S2 (UNI - Uniform Voting):* Each neighbor contributes an equal vote regardless of similarity score. This democratic approach treats all neighbors equally and may be more robust to outliers.

== Overall Results

The comprehensive evaluation tested 7 feature extractors × 9 environment strategies (E1, E2, 7×E3) × 2 aggregation methods = 126 total configurations. Results are organized by feature extractor.

=== Deep Learning Methods

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, left, left, right, right),
    table.header(
      [*Feature Extractor*], [*Environment*], [*Aggregation*], [*Accuracy*], [*Correct/Total*],
    ),
    
    // ResNet50
    [ResNet50], [E1], [S1], [64.58%], [31/48],
    [ResNet50], [E1], [S2], [64.58%], [31/48],
    [ResNet50], [E2], [S1], [66.67%], [32/48],
    [ResNet50], [E2], [S2], [66.67%], [32/48],
    [ResNet50], [E3_CREA], [S1], [45.83%], [22/48],
    [ResNet50], [E3_CREA], [S2], [45.83%], [22/48],
    [ResNet50], [E3_CYA], [S1], [29.17%], [14/48],
    [ResNet50], [E3_CYA], [S2], [29.17%], [14/48],
    [ResNet50], [E3_CYA30], [S1], [54.17%], [26/48],
    [ResNet50], [E3_CYA30], [S2], [54.17%], [26/48],
    [ResNet50], [E3_CYAS], [S1], [33.33%], [16/48],
    [ResNet50], [E3_CYAS], [S2], [35.42%], [17/48],
    [ResNet50], [E3_DG18], [S1], [39.58%], [19/48],
    [ResNet50], [E3_DG18], [S2], [39.58%], [19/48],
    [ResNet50], [E3_MEA], [S1], [43.75%], [21/48],
    [ResNet50], [E3_MEA], [S2], [41.67%], [20/48],
    [ResNet50], [E3_YES], [S1], [54.17%], [26/48],
    [ResNet50], [E3_YES], [S2], [47.92%], [23/48],
    
    // MobileNetV2
    [MobileNetV2], [E1], [S1], [54.17%], [26/48],
    [MobileNetV2], [E1], [S2], [45.83%], [22/48],
    [MobileNetV2], [E2], [S1], [52.08%], [25/48],
    [MobileNetV2], [E2], [S2], [52.08%], [25/48],
    [MobileNetV2], [E3_CREA], [S1], [25.00%], [12/48],
    [MobileNetV2], [E3_CREA], [S2], [27.08%], [13/48],
    [MobileNetV2], [E3_CYA], [S1], [25.00%], [12/48],
    [MobileNetV2], [E3_CYA], [S2], [25.00%], [12/48],
    [MobileNetV2], [E3_CYA30], [S1], [31.25%], [15/48],
    [MobileNetV2], [E3_CYA30], [S2], [31.25%], [15/48],
    [MobileNetV2], [E3_CYAS], [S1], [27.08%], [13/48],
    [MobileNetV2], [E3_CYAS], [S2], [27.08%], [13/48],
    [MobileNetV2], [E3_DG18], [S1], [37.50%], [18/48],
    [MobileNetV2], [E3_DG18], [S2], [39.58%], [19/48],
    [MobileNetV2], [E3_MEA], [S1], [31.25%], [15/48],
    [MobileNetV2], [E3_MEA], [S2], [31.25%], [15/48],
    [MobileNetV2], [E3_YES], [S1], [52.08%], [25/48],
    [MobileNetV2], [E3_YES], [S2], [52.08%], [25/48],
    
    // EfficientNetV2B0
    [EfficientNetV2B0], [E1], [S1], [62.50%], [30/48],
    [EfficientNetV2B0], [E1], [S2], [62.50%], [30/48],
    [EfficientNetV2B0], [E2], [S1], [52.08%], [25/48],
    [EfficientNetV2B0], [E2], [S2], [52.08%], [25/48],
    [EfficientNetV2B0], [E3_CREA], [S1], [35.42%], [17/48],
    [EfficientNetV2B0], [E3_CREA], [S2], [39.58%], [19/48],
    [EfficientNetV2B0], [E3_CYA], [S1], [25.00%], [12/48],
    [EfficientNetV2B0], [E3_CYA], [S2], [25.00%], [12/48],
    [EfficientNetV2B0], [E3_CYA30], [S1], [45.83%], [22/48],
    [EfficientNetV2B0], [E3_CYA30], [S2], [45.83%], [22/48],
    [EfficientNetV2B0], [E3_CYAS], [S1], [22.92%], [11/48],
    [EfficientNetV2B0], [E3_CYAS], [S2], [22.92%], [11/48],
    [EfficientNetV2B0], [E3_DG18], [S1], [29.17%], [14/48],
    [EfficientNetV2B0], [E3_DG18], [S2], [29.17%], [14/48],
    [EfficientNetV2B0], [E3_MEA], [S1], [39.58%], [19/48],
    [EfficientNetV2B0], [E3_MEA], [S2], [39.58%], [19/48],
    [EfficientNetV2B0], [E3_YES], [S1], [52.08%], [25/48],
    [EfficientNetV2B0], [E3_YES], [S2], [52.08%], [25/48],
  ),
  caption: [Deep learning methods performance. ResNet50 achieved best results (66.67% with E2).],
)

#pagebreak()

=== Traditional Computer Vision Methods

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, left, left, right, right),
    table.header(
      [*Feature Extractor*], [*Environment*], [*Aggregation*], [*Accuracy*], [*Correct/Total*],
    ),
    
    // ColorHistogram
    [ColorHistogram], [E1], [S1], [62.50%], [30/48],
    [ColorHistogram], [E1], [S2], [52.08%], [25/48],
    [ColorHistogram], [E2], [S1], [60.42%], [29/48],
    [ColorHistogram], [E2], [S2], [60.42%], [29/48],
    [ColorHistogram], [E3_CREA], [S1], [41.67%], [20/48],
    [ColorHistogram], [E3_CREA], [S2], [41.67%], [20/48],
    [ColorHistogram], [E3_CYA], [S1], [33.33%], [16/48],
    [ColorHistogram], [E3_CYA], [S2], [33.33%], [16/48],
    [ColorHistogram], [E3_CYA30], [S1], [16.67%], [8/48],
    [ColorHistogram], [E3_CYA30], [S2], [16.67%], [8/48],
    [ColorHistogram], [E3_CYAS], [S1], [29.17%], [14/48],
    [ColorHistogram], [E3_CYAS], [S2], [29.17%], [14/48],
    [ColorHistogram], [E3_DG18], [S1], [45.83%], [22/48],
    [ColorHistogram], [E3_DG18], [S2], [45.83%], [22/48],
    [ColorHistogram], [E3_MEA], [S1], [39.58%], [19/48],
    [ColorHistogram], [E3_MEA], [S2], [39.58%], [19/48],
    [ColorHistogram], [E3_YES], [S1], [43.75%], [21/48],
    [ColorHistogram], [E3_YES], [S2], [43.75%], [21/48],
    
    // ColorHistogramHS - Best Overall
    [*ColorHistogramHS*], [*E1*], [*S1*], [*60.42%*], [*29/48*],
    [*ColorHistogramHS*], [*E1*], [*S2*], [*52.08%*], [*25/48*],
    [*ColorHistogramHS*], [*E2*], [*S1*], [*75.00%*], [*36/48*],
    [*ColorHistogramHS*], [*E2*], [*S2*], [*75.00%*], [*36/48*],
    [ColorHistogramHS], [E3_CREA], [S1], [43.75%], [21/48],
    [ColorHistogramHS], [E3_CREA], [S2], [43.75%], [21/48],
    [ColorHistogramHS], [E3_CYA], [S1], [29.17%], [14/48],
    [ColorHistogramHS], [E3_CYA], [S2], [29.17%], [14/48],
    [ColorHistogramHS], [E3_CYA30], [S1], [12.50%], [6/48],
    [ColorHistogramHS], [E3_CYA30], [S2], [12.50%], [6/48],
    [ColorHistogramHS], [E3_CYAS], [S1], [33.33%], [16/48],
    [ColorHistogramHS], [E3_CYAS], [S2], [35.42%], [17/48],
    [ColorHistogramHS], [E3_DG18], [S1], [31.25%], [15/48],
    [ColorHistogramHS], [E3_DG18], [S2], [33.33%], [16/48],
    [ColorHistogramHS], [E3_MEA], [S1], [39.58%], [19/48],
    [ColorHistogramHS], [E3_MEA], [S2], [39.58%], [19/48],
    [ColorHistogramHS], [E3_YES], [S1], [39.58%], [19/48],
    [ColorHistogramHS], [E3_YES], [S2], [39.58%], [19/48],
    
    // HOG
    [HOG], [E1], [S1], [39.58%], [19/48],
    [HOG], [E1], [S2], [37.50%], [18/48],
    [HOG], [E2], [S1], [45.83%], [22/48],
    [HOG], [E2], [S2], [45.83%], [22/48],
    [HOG], [E3_CREA], [S1], [27.08%], [13/48],
    [HOG], [E3_CREA], [S2], [29.17%], [14/48],
    [HOG], [E3_CYA], [S1], [18.75%], [9/48],
    [HOG], [E3_CYA], [S2], [20.83%], [10/48],
    [HOG], [E3_CYA30], [S1], [22.92%], [11/48],
    [HOG], [E3_CYA30], [S2], [22.92%], [11/48],
    [HOG], [E3_CYAS], [S1], [20.83%], [10/48],
    [HOG], [E3_CYAS], [S2], [20.83%], [10/48],
    [HOG], [E3_DG18], [S1], [33.33%], [16/48],
    [HOG], [E3_DG18], [S2], [33.33%], [16/48],
    [HOG], [E3_MEA], [S1], [29.17%], [14/48],
    [HOG], [E3_MEA], [S2], [29.17%], [14/48],
    [HOG], [E3_YES], [S1], [31.25%], [15/48],
    [HOG], [E3_YES], [S2], [31.25%], [15/48],
    
    // Gabor
    [Gabor], [E1], [S1], [35.42%], [17/48],
    [Gabor], [E1], [S2], [35.42%], [17/48],
    [Gabor], [E2], [S1], [27.08%], [13/48],
    [Gabor], [E2], [S2], [27.08%], [13/48],
    [Gabor], [E3_CREA], [S1], [27.08%], [13/48],
    [Gabor], [E3_CREA], [S2], [25.00%], [12/48],
    [Gabor], [E3_CYA], [S1], [20.83%], [10/48],
    [Gabor], [E3_CYA], [S2], [20.83%], [10/48],
    [Gabor], [E3_CYA30], [S1], [22.92%], [11/48],
    [Gabor], [E3_CYA30], [S2], [20.83%], [10/48],
    [Gabor], [E3_CYAS], [S1], [14.58%], [7/48],
    [Gabor], [E3_CYAS], [S2], [16.67%], [8/48],
    [Gabor], [E3_DG18], [S1], [39.58%], [19/48],
    [Gabor], [E3_DG18], [S2], [39.58%], [19/48],
    [Gabor], [E3_MEA], [S1], [18.75%], [9/48],
    [Gabor], [E3_MEA], [S2], [20.83%], [10/48],
    [Gabor], [E3_YES], [S1], [31.25%], [15/48],
    [Gabor], [E3_YES], [S2], [31.25%], [15/48],
  ),
  caption: [Traditional computer vision methods. ColorHistogramHS E2 achieved best overall accuracy (75%).],
)

== Detailed Case Studies

Five representative configurations are analyzed to demonstrate performance across different conditions.

=== Case 1: Best Overall Performance - ColorHistogramHS E2 AVG (75.00%)

#figure(
  image("assets/results/comprehensive_k7_NoSib/ColorHistogramHS_E2_AVG/confusion_matrix_20251113_153320.png", width: 70%),
  caption: [ColorHistogramHS E2 AVG confusion matrix - best overall accuracy (36/48 correct).],
)

Achieved highest accuracy by combining Hue-Saturation color features with cross-environment queries. Strong diagonal indicates consistent predictions across most species. Cross-environment generalization (E2: 75%) significantly outperformed same-environment queries (E1: 60.42%).

#figure(
  grid(
    columns: 2,
    gutter: 10pt,
    image("assets/results/comprehensive_k7_NoSib/ColorHistogramHS_E2_AVG/false_predictions/DTO_148-C8_3efbd33ab2e8_pred_Penicillium_neoechin.jpg", width: 100%),
    image("assets/results/comprehensive_k7_NoSib/ColorHistogramHS_E2_AVG/false_predictions/DTO_158-D1_5966b68c1ce8_pred_Penicillium_freii.jpg", width: 100%),
  ),
  caption: [Example false predictions from best performer showing remaining challenges.]
)

=== Case 2: Same Environment Strategy - ColorHistogramHS E1 AVG (60.42%)

#figure(
  image("assets/results/comprehensive_k7_NoSib/ColorHistogramHS_E1_AVG/confusion_matrix_20251113_153306.png", width: 70%),
  caption: [ColorHistogramHS E1 AVG confusion matrix (29/48 correct).],
)

When querying within the same environment (E1), accuracy dropped to 60.42%, suggesting that environment-specific training reduces generalization. However, this still outperformed most other feature extractors.

#figure(
  grid(
    columns: 2,
    gutter: 10pt,
    image("assets/results/comprehensive_k7_NoSib/ColorHistogramHS_E1_AVG/false_predictions/DTO_163-I2_5b1f88ff7021_pred_Penicillium_aurantio.jpg", width: 100%),
    image("assets/results/comprehensive_k7_NoSib/ColorHistogramHS_E1_AVG/false_predictions/DTO_469-I5_6fd969b296fd_pred_Penicillium_polonicu.jpg", width: 100%),
  ),
  caption: [E1 strategy false predictions showing single-environment limitations.]
)

=== Case 3: Deep Learning Comparison - ResNet50 E1 AVG (64.58%)

#figure(
  image("assets/results/comprehensive_k7_NoSib/ResNet50_E1_AVG/confusion_matrix_20251113_153033.png", width: 70%),
  caption: [ResNet50 E1 AVG confusion matrix (31/48 correct).],
)

Best-performing deep learning method achieved 64.58% accuracy with E1 strategy. Despite pre-training on ImageNet, ResNet50 did not surpass traditional color histograms, suggesting deep features are less discriminative than color information for fungal colonies.

#figure(
  grid(
    columns: 2,
    gutter: 10pt,
    image("assets/results/comprehensive_k7_NoSib/ResNet50_E1_AVG/false_predictions/DTO_148-C8_3efbd33ab2e8_pred_Penicillium_neoechin.jpg", width: 100%),
    image("assets/results/comprehensive_k7_NoSib/ResNet50_E1_AVG/false_predictions/DTO_163-I2_5b1f88ff7021_pred_Penicillium_polonicu.jpg", width: 100%),
  ),
  caption: [Deep learning false predictions showing transfer learning limitations.]
)

=== Case 4: Challenging Environment - ColorHistogramHS E3_CYA AVG (29.17%)

#figure(
  image("assets/results/comprehensive_k7_NoSib/ColorHistogramHS_E3_CYA_AVG/confusion_matrix_20251113_153335.png", width: 70%),
  caption: [ColorHistogramHS E3_CYA AVG confusion matrix (14/48 correct).],
)

CYA environment showed poorest performance (29.17%), representing 45.83 percentage point drop from E2. Weak diagonal and scattered predictions suggest CYA medium produces similar colony morphology across species, limiting color-based discrimination.

#figure(
  grid(
    columns: 2,
    gutter: 10pt,
    image("assets/results/comprehensive_k7_NoSib/ColorHistogramHS_E3_CYA_AVG/false_predictions/DTO_148-C8_18f492827651_pred_Penicillium_aurantio.jpg", width: 100%),
    image("assets/results/comprehensive_k7_NoSib/ColorHistogramHS_E3_CYA_AVG/false_predictions/DTO_470-I9_359ce680911d_pred_Penicillium_viridica.jpg", width: 100%),
  ),
  caption: [CYA environment false predictions showing challenging growth medium effects.]
)

=== Case 5: Aggregation Strategy - ColorHistogramHS E2 UNI (75.00%)

#figure(
  image("assets/results/comprehensive_k7_NoSib/ColorHistogramHS_E2_UNI/confusion_matrix_20251113_153326.png", width: 70%),
  caption: [ColorHistogramHS E2 UNI confusion matrix - identical accuracy with uniform voting (36/48 correct).],
)

Uniform voting (S2/UNI) matched weighted voting (S1/AVG) at 75% accuracy, confirming aggregation strategy has minimal impact when feature quality is consistent. Similar confusion patterns across both methods validate robustness of ColorHistogramHS features.

#figure(
  grid(
    columns: 2,
    gutter: 10pt,
    image("assets/results/comprehensive_k7_NoSib/ColorHistogramHS_E2_UNI/false_predictions/DTO_148-C8_3efbd33ab2e8_pred_Penicillium_neoechin.jpg", width: 100%),
    image("assets/results/comprehensive_k7_NoSib/ColorHistogramHS_E2_UNI/false_predictions/DTO_158-D1_5966b68c1ce8_pred_Penicillium_freii.jpg", width: 100%),
  ),
  caption: [Aggregation comparison showing identical failure cases across voting strategies.]
)

== Key Findings

*Best Configuration:* ColorHistogramHS E2 AVG/UNI achieved 75% accuracy (36/48 correct), outperforming all 126 tested combinations. Hue-Saturation color space proved most discriminative for fungal identification.

*Environment Strategy Impact:* Cross-environment queries (E2) consistently outperformed same-environment (E1) and single-environment (E3) approaches. E2 enabled better generalization across growth conditions.

*Environment Difficulty:* Performance varied by medium - CYA/CYAS most challenging (12.5-35.42%), YES/CYA30 most discriminative (39.58-54.17%). CYA's uniform colony morphology limits color-based discrimination.

*Traditional vs. Deep Learning:* Color histograms (ColorHistogramHS: 75%, ColorHistogram: 62.5%) outperformed deep learning (ResNet50: 66.67%, EfficientNetV2B0: 62.5%). Color distribution more informative than learned features for fungal colonies.

*Aggregation Strategy:* Weighted (AVG) vs. uniform (UNI) voting showed minimal difference (typically <2% points). For consistent features like ColorHistogramHS, aggregation method is negligible.
