# DL-CancerLncRNA

<p align="center">

<a >
    <img src='images/logo.png'  width="400"/>
</a>

</p>


## Objective
The present project consists on the development of an AI method based on deep-learning to associate long non-coding RNAs (lncRNAs) to cancer types and biological functions. 

Long non-coding RNAs (**LncRNAs**) are transcripts >200 nt that do not code for proteins. There are ~40 000 - 60 000 lncRNAs in the human genome. However, up to now we only have information for much less than 1000. Several of them are implicated in important cell processes and maladies such as cancer.
Experimental characterisation of lncRNAs it’s a long process. **Bioinformatics approaches are urgently needed** to identify the lncRNAs of interest for **clinical applications**.
Tons of data for their study are already available, and many of them **are public**!
In this project we collected some of the publicly available data such as:

- LncRNAs sequences → https://www.gencodegenes.org/ https://www.ensembl.org/index.html 

- LncRNA expression data across ~1500 samples of cancer → https://dcc.icgc.org/pcawg 

- LncRNAs and cancer associations →  http://www.bio-bigdata.com/lnc2cancer/  https://www.gold-lab.org/clc

## Applications
The proposed model can be useful to **identity lncRNAs implicated in cancer** for further study and **help to detect new possible therapeutic solutions**. LncRNAs are in fact very promising therapeutic targets because their expression pattern is tissue specific **reducing the risk of off-targets and toxicity in cancer treatments**.
This approach can be explored with other **publicly available** datasets:
- Pan-Cancer Atlas initiative comparing 33 tumor types profiled by TCGA: https://gdc.cancer.gov/about-data/publications/pancanatlas
- The Genotype-Tissue Expression (GTEx) project: https://www.gtexportal.org/home/
- Cancer Cell Line Encyclopedia (CCLE): https://sites.broadinstitute.org/ccle/

## Demo

To start the API, use the following command : 
```shell
make deploy_api
```
And then you can enter the gene id, which is then going to return the top 3 cancers that our model predict. 
It will then generate barcharts for either the cancer or the functions associated. 

<p align="center">

<a >
    <img src='images/demo.gif'  width="800"/>
</a>

</p>





## Installation 

To run the code and do the learning process, one should use the following command : 

```shell
make eval MODEL=<MODEL_NAME> ARGS="ARG1 VALUE1 ARG2 VALUE2"
```
with <MODEL_NAME> the name of the model to use (usually a class in the `models` directory). 
If one wants to add arguments to the model, use it in the `ARGS` parameter.


## Data 

### Description

DESCRIPTION OF THE WAY WE COLLECTED THE DATA 

### Visualisation 

We have collected three main datasets : one with the sequences of the RNAs, one with the expressions and the last one with both of them. 

Here is a visualisation of the distribution of the labels : 

| Sequences | Expressions |
|---| --- |
| ![](/images/sequences_cancer.png) | ![](/images/expressions_cancer.png) |


The final dataset with both of the features of the RNA has the following proportion between the different representations : 

<p align="center">
<a >
    <img src='./images/intersection.png'  width="400"/>
</a>
</p>

To get the plots of the visualisations, use the following command : 

```shell
make viz
```

## Model 

We tried different models for both the functional and sequential data. 

### Functional 

# TODO (CONSTANCE, LILIA)

### Sequential

For the sequential part, we only have as inputs a sequence of nucleotide for each RNA.

Therefore, we tried to convert those sequences into vectors, that give more information than just the nucleotids. 

Here is a list of the preprocessing we tried : 

- `One-hot-encoding` : we convert each nucleotid into a vector of size 4, with only one 1 value. 
- `K-mer` : we count the occurence of the k-mer in the sequence. We then used 4-mer to try to learn different model 
- # TODO : LOIC

For the models, we used either pytorch or keras to train the models. 

In pytorch, we tried : 
- `GRU` : with either one-hot-encoding or 4-mer as preprocessing. We used either 128 or 256 as hidden dimensions. 
- `LSTM` : with either one-hot-encoding or 4-mer as preprocessing. We also used either 128 or 256 hidden dimensions.

In keras, we tried : 
- `CNN` : #TODO (Loic)



## Teams 

- TEAM Leader: Constance CREUX - [IBISC](https://www.ibisc.univ-evry.fr/)
- Lilia ESTRADA - [Institut Curie](https://institut-curie.org/)
- Loïc OMNES - [IBISC](https://www.ibisc.univ-evry.fr/)
- Clément BERNARD - [Telecom Paris / Eurecom / Polytechnique Montréal]