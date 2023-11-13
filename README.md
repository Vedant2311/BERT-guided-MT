# BERT-guided-MT
Utilizing expert pre-trained systems like BERT to improve the performance of low resource NMT
TODO: Improve the README as we go along the project

## Getting raw data for Nepali and English
We use two publically available corpora for getting the training data
1) Bible Corpus (62K sentences, 1.5M tokens) https://link.springer.com/article/10.1007/s10579-014-9287-y
2) OPUS GNOME/KDE/Ubuntu (495K sentences, 2M tokens) https://opus.nlpl.eu

We download and combine the raw training data from these sources by using the script *combineTrain.py*. Upon analysing this dataset, we observed that there were certain noisy training pairs and repetitions in this overall training set. To get combat any possible training detorioration due to this unclean data, we perform some cleaning steps which are included in *cleanTrain.py*

We use the FLORES-200 evaluation dataset to get the dev and the test set for Nepali and English. This data can be downloaded here - https://github.com/facebookresearch/flores/tree/main/previous_releases/floresv1.

All the train, dev, and test data, along with the relevant scripts, are present in the **dataset** folder. Note that the language code for English is en_XX and the language code for Nepali is ne_NP.

## Performing tokenization before fine-tuning the models
TODO
