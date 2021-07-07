# Project Description

This is a multiclass classification problem.  
We want to correctly predict each cancer class while using the multiomics views available.  
The desirable behaviour of the model is to be __robust__ even though some views are not availble for the patient (which is a strong real case); to use all the information available from the views and to be the most accurate possible.

## Datasets

We have 33 cancers (all in TCGA) retrieve from [https://xenabrowser.net/datapages/?cohort=TCGA%20Pan-Cancer%20(PANCAN)&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443].  
There is various views available for all of them: __cnv__, __methyl__, __miRNA__, __RNA__ (iso and normal), __protein__,  __pathway activity__, __signatures__, __snp and indel__.
Here we decide to go with the 4 principal views: __cnv__, __methyl__, __miRNA__, __RNAiso__.  
It is worth mentionning that there is some view(s) that is(are) not available for some patients i.e some patients don't have all the 4 views available.
The examples we consider have at least 2 views in the final set.  

## Model

1. Transformer model with attention head
2. Optimize with the accuracy while using Optuna

## Experimentations  

There is 3 types of experimentation that were done:

1. Run the model on each views independently
2. Run the model on all the views. When the views are not available for the patient, it is field with __0__ in the matrices.
    1. Example: if we have a patient that only have 2 views available. We will still have a 4*N matrices with the 2 rows set at 0.
3. Run the model on __augmentation dataset views__. This consist in building (randomly) a different combination of views for each patient. 
    1. Example: if we have a patient who has all the 4 views, we can sample a dataset containing all the view, or just 3 or 2 or even one. This is the part that is simulating the randomness of availability of data (in our domain) per patients.

4. Run the model on  __augmentation dataset views__ and combined another decoder for learning the missing representation. I called it the __multimodal__ model since it's optimizing 2 tasks Simultaneously.
    1. Example: if we have a representation of the patient, we will encode-decode for the prediction task while learning a decoder for optimizing the reconstruction of the real dataset. 2 losses are being Optimized.

For the experimentation, we made a feature selection (all the feature space could not fit in) using __median_abs_deviation__.  
We arbitrary selected to test our model on 3 types: __NbFeatures = 2000__; __NbFeatures = 5000__; and __NbFeatures = 10000__.

## Results  

| Data        | Metrics           | Notes           |
| ------------- |:-------------:| -------------:|
| CNV     | {"acc": __53.469__, "prec": 58.925, "rec": 53.469, "f1_score": 55.105} | |
| Methyl | {"acc": __88.444__, "prec": 89.856, "rec": 88.444, "f1_score": 88.775} | |
| miRNA | {"acc": __93.553__, "prec": 93.554, "rec": 93.553, "f1_score": 93.429} | |
| RNAiso | {"acc": __90.758__, "prec": 92.057, "rec": 90.758, "f1_score": 90.874} | |
| all_2000_data| {"acc": __90.275__, "prec": 91.132, "rec": 90.275, "f1_score": 90.508} | Hp combinations were different but the same within the experiment|
| all_5000_data| {"acc": __92.141__, "prec": 92.936, "rec": 92.141, "f1_score": 92.384} | Hp combinations were different but the same within the experiment|
| all_10000_data| {"acc": __95.236__, "prec": 95.66, "rec": 95.236, "f1_score": 95.341} | Hp combinations were different but the same within the experiment|
| all_2000_data_augmentation| {"acc": __89.391__, "prec": 89.971, "rec": 89.391, "f1_score": 89.559} | Doesnt help|
| all_5000_data_augmentation| {"acc": __87.819__, "prec": 88.961, "rec": 87.819, "f1_score": 88.048}| Doesnt help|
| all_10000_data_augmentation| {"acc": __91.896__, "prec": 92.384, "rec": 91.896, "f1_score": 91.962} | Doesnt help|
| all_2000_data_augmentation_multimodal| {"acc": __89.686__, "prec": 90.628, "rec": 89.686, "f1_score": 89.946} | Idk if i'm doing the right thing but it seems it doesnt help|
| all_5000_data_augmentation_multimodal| {"acc": __90.864__, "prec": 91.707, "rec": 90.864, "f1_score": 91.129} | Idk if i'm doing the right thing but it seems it doesnt help|
| all_10000_data_augmentation_multimodal| {"acc": __86.542__, "prec": 88.08, "rec": 86.542, "f1_score": 86.917} | Idk if i'm doing the right thing but it seems it doesnt help |