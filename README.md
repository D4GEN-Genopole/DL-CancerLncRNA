# DL-CancerLncRNA
AI Technologies

TEAM Leader: Constance CREUX - [IBISC](https://www.ibisc.univ-evry.fr/)

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


# CLI 

Pour run le code, il faut utiliser les commandes suivantes : 
```shell
make eval MODEL=<MODEL_NAME> ARGS="ARG1 VALUE1 ARG2 VALUE2"
```
avec <MODEL_NAME> le nom de la classe du modèle, et les arguments donnés dans `ARGS`. 
