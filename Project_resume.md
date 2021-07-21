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

## Stories angles

### Observation 1

The more data we got the better the results get for the normal and the data augmentation experimentation.
The only difference is on the multitask problem.
But it might be due to the facts that some of the hps combination did not run for all the dataset.
**Questions**

1. Should we add more features? if yes where will we stop? i mean i can go further (15k or 20k) assuming it can go in the memory but in that path (assuming the trends is sustained) where will we stop?

2. Should we present those results or just pick already the 10k data and after say in the supplementarry data why we picked 10k? based on those results...

### Observation 2
Here we will present a table about the attention scores (i.e the views that was more emphasis for each experimentation above)

| Cancer        | all_2000_data | all_5000_data | all_10000_data |all_2000_data_augmentation | all_5000_data_augmentation | all_10000_data_augmentation |all_2000_data_augmentation_multimodal | all_5000_data_augmentation_multimodal | all_10000_data_augmentation_multimodal |
| ------------- |:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
|__ACC__| cnv-cnv(0.57) | | minra-methyl(0.69)|rna-cnv(0.4) mirna-cnv(0.38) methyl-mirna(0.38)| | mirna-methyl(0.48)| | | | 
|__BLCA__| rna-cnv(0.44) methyl-cnv(0.42) cnv-cnv(0.35) mirna-cnv(0.31)| |minra-methyl(0.52) | rna-cnv(0.46) mirna-cnv(0.45)| | methyl-methyl(0.41) rna-cnv(0.39)| | | | 
|__BRCA__| cnv-cnv(0.42) mirna-cnv(0.42) rna-cnv(0.37) methyl-cnv(0.34)| | minra-methyl(0.39) methyl-cnv(0.35) |mirna-cnv(0.47) | | methyl-rna(0.37) too much value around 0.35| | | | 
|__CESC__| mirna-methyl(0.47) methyl-cnv(0.42) rna-cnv(0.41) |  | minra-methyl(0.44)| mirna-cnv(0.39)| |methyl-methyl(0.52) | | | | 
|__CHOL__| mirna-methyl(0.66) | | minra-methyl(0.95) | cnv-methyl(0.75)| | cnv-methyl(0.55)| | | | 
|__COAD__| cnv-cnv(0.39) | | minra-methyl(0.39) methyl-methyl(0.39) rna-methyl(0.38) |mirna-cnv(0.5) methyl-cnv(0.49) | | methyl-cnv(0.4) rna-methyl(0.39) | | | | 
|__DLBC__| methyl-mirna(0.52) | |methyl-cnv(0.54) | rna-cnv(0.66)| | rna-rna(0.51)| | | | 
|__ESCA__|mirna-cnv(0.45) | | minra-methyl(0.46) cnv-cnv(0.4)|mirna-cnv(0.55) | | rna-cnv(0.49)| | | | 
|__GBM__| rna-cnv(0.67) | | rna-methyl(0.7) |rna-cnv(0.73) | | rna-rna(0.59)| | | | 
|__HNSC__|rna-cnv(0.34) methyl-mirna(0.34) methyl-cnv(0.33)  too much closer to 0.34| | minra-methyl(0.47) methyl-cnv(0.45)| mirna-cnv(0.52)| | mirna-methyl(0.48)| | | | 
|__KICH__| rna-cnv(0.6) | | minra-methyl(0.53)| cnv-rna(0.62)| | cnv-methyl(0.56) cnv-methyl(0.55)| | | | 
|__KIRC__|cnv-cnv(0.39) methyl-cnv(0.37) mirna-cnv(0.33) rna-cnv(0.29) | | minra-methyl(0.38) too much around 0.36|rna-cnv(0.53) | | cnv-cnv(0.47)| | | | 
|__KIRP__| methyl-methyl(0.32)  too much closer to 0.32 | |minra-methyl(0.51) |methyl-methyl(0.42) | | mirna-methyl(0.61)| | | | 
|__LAML__| mirna-methyl(0.38) rna-cnv(0.37) cnv-methyl(0.36) methyl-methyl(0.34)| | mirna-methyl(0.54) | cnv-methyl(0.47) methyl-methyl(0.45)| | methyl-rna(0.57)| | | | 
|__LGG__|mirna-cnv(0.38) cnv-cnv(0.32) rna-methyl(0.32)| |minra-methyl(0.66)|mirna-cnv(0.47) | |mirna-methyl(0.51) | | | | 
|__LIHC__| mirna-cnv(0.42) methyl-methyl(0.4) | |minra-methyl(0.53)|mirna-cnv(0.41) rna-cnv(0.41) | | rna-cnv(0.42) too much value around 0.38| | | | 
|__LUAD__| mirna-cnv(0.49) cnv-cnv(0.42)| | minra-methyl(0.51) |mirna-cnv(0.48) | | cnv-cnv(0.44) methyl-cnv(0.44) rna-cnv(0.42)| | | | 
|__LUSC__| rna-cnv(0.45) cnv-cnv(0.42) mirna-cnv(0.42) | | mirna-rna(0.52) | mirna-cnv(0.64)| | methyl-cnv(0.48) rna-cnv(0.46)| | | | 
|__MESO__| mirna-methyl(0.49) | |rna-cnv(0.49) minra-methyl(0.48) | mirna-cnv(0.63)| | mirna-methyl(0.59) | | | | 
|__OV__| methyl-cnv(0.47) | | methyl-rna(0.52) cnv-rna(0.5)| mirna-cnv(0.6)| | mirna-rna(0.59) | | | | 
|__PAAD__| methyl-cnv(0.51) rna-cnv(0.44) | |minra-methyl(0.48) | mirna-ccnv(0.41) | | methyl-cnv(0.4) rna-methyl(0.4) | | | | 
|__PCPG__|rna-methyl(0.43) mirna-methyl(0.4) cnv-cnv(0.39) | |minra-methyl(0.61) |rna-methyl(0.41) methyl-methyl(0.4) | |mirna-methyl(0.59) | | | | 
|__PRAD__|rna-cnv(0.39) methyl-cnv(0.36) cnv-cnv(0.35) mirna-cnv(0.34) | | minra-methyl(0.59)| mirna-cnv(0.5) rna-cnv(0.5) | |rna-methyl(0.51) mirna-methyl(0.5) cnv-methyl(0.48) | | | | 
|__READ__|mirna-cnv(0.44) cnv-cnv(0.43) rna-cnv(0.42) methyl-cnv(0.39) | | methyl-mirna(0.39)too much value around 0.35| mirna-mirna(0.36)too much value around 0.33 | | methyl-cnv(0.43) too much value around 0.39|  | | | 
|__SARC__|mirna-cnv(0.44) rna-cnv(0.41) methyl-cnv(0.34) cnv-ccnv(0.32) | | minra-methyl(0.6)| methyl-cnv(0.4) mirna-cnv(0.38) | | mirna-methyl(0.46)| | | | 
|__SKCM__|methyl-methyl(0.41) mirna-cnv(0.36) mirna-mirna(0.34| | minra-methyl(0.57) | cnv-methyl(0.44) methyl-cnv(0.4) mirna-cnv(0.4)| | mirna-methyl(0.4) rna-methyl(0.39) cnv-meethyl(0.37)| | | | 
|__STAD__|cnv-cnv(0.42) rna-cnv(0.36) mirna-cnv(0.33) methyl-cnv(0.3) | | minra-methyl(0.53)|mirna-cnv(0.5) | | cnv-cnv(0.49)| | | | 
|__TGCT__| mirna-cnv(0.39) cnv-cnv(0.36) methyl-cnv(0.33) | | methyl-cnv(0.42) mirna-methyl(0.4)|mirna-cnv(0.44) | | rna-cnv(0.45) mirna-methyl(0.44) cnv-methyl(0.44)| | | | 
|__THCA__| mirna-methyl(0.34) mirna-mirna(0.29) cnv-mirna(0.29) methyl-mirna(0.28)| | mirna-methyl(0.64)|methyl-methyl(0.45) cnv-methyl(0.42) | |methyl-methyl(0.66) | | | | 
|__THYM__| methyl-methyl(0.55) rna-cnv(0.41) mirna-mirna(0.4) | | mirna-methyl(0.58) cnv-methyl(0.52) | methyl-methyl(0.52) | | methyl-methyl(0.46)| | | | 
|__UCEC__| rna-cnv(0.48) methyl-cnv(0.42) mirna-mirna(0.39)| | mirna-methyl(0.54)|mirna-cnv(0.51) methyl-cnv(0.5)| | methyl-methyl(0.46)| | | | 
|__UCS__| rna-cnv(0.52) methyl-cnv(0.52) cnv-methyl(0.46) | | mirna-methyl(0.61)| mirna-cnv(0.7)| | methyl-methyl(0.48)| | | | 
|__UVM__|rna-methyl(0.53) mirna-miran(0.43) methyl-mirna(0.46) | | mirna-methyl(0.63)| mirna-cnv(0.67)| |mirna-methyl(0.56) rna-cnv(0.54) cnv-methyl(0.5)| | | | 

1. Normal experimentation
    1. __READ__: comportement similaire i.e even score (l'algo regarde tout ensemble quelque soit la taille)
    2. __10k vs 2k__: on met clairement plus l'emphase sur la combinaison mirna-methyl pour la plus part d'entre eux. Serait-ce un comportement du à la présence de plus de feautures?
    3,. Certains cancer il n'y a pas d'attention particulière vu que les scores ne sont pas drastiquement différents. On peut dire que les models regardent tout en meme temps. Ces cancers sont: __READ__, __TGCT__, __THYM__,__KIRC__,__HNSC__   (__BLCA__, __BRCA__, __PRAD__, __PCPG__, __LAML__ si on 
    considere 2k seulement)

2. Data Augmentation
    1. Je ne vois pas un pattern ressortir ici entre les expériences à part que la vue methyle st plus regardé en combinaison avec mirna dans le cas __10k data augmentation__ versus le __2k data augmentation__.

3. Data Augmentation Multimodal (pas fait vu que les résultats sont so-so pour le moment)

4. Suprrenement je trouve que la vue cnv (en combination avec autre chose) est très regardée. suivi de methyl (ca c'est pas surprenant) mais rna est quasiment pas regardé (bizarre aussi).

5. Je vais relancer les expérimentations et essayer de nouveauz hps pour voir