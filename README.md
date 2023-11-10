# BERT-guided-MT
Utilizing expert pre-trained systems like BERT to improve the performance of low resource NMT
TODO: Improve the README as we go along the project

## References to be cited in the paper
- NepBERTa - BERT based NLU model exclusively for Nepali
https://aclanthology.org/2022.aacl-short.34.pdf

- FLORES evaluation dataset for Ne-En (test + dev)
https://aclanthology.org/D19-1632/

- Training data for Ne-En
1) Bible Corpus (62K sentences, 1.5M tokens) https://link.springer.com/article/10.1007/s10579-014-9287-y
2) GNOME/KDE/Ubuntu (495K sentences, 2M tokens) https://opus.nlpl.eu

- Augmenting mBART with Back Translation and Transfer Learning to survey benefits in low resource NMT. Demonstrated benefits on Ne-En with Back Translation (Could be one of our baselines)
https://arxiv.org/pdf/2304.07869.pdf

- Incorporating BERT into NMT (Demonstrated improvements in high resource languages like English, German, and French. Also did not use heavier models like mBART. Used two techniques - Initialized embeddings and Used the output embeddings as context-aware embeddings for NMT)
https://openreview.net/attachment?id=Hyl7ygStwB&name=original_pdf

- Survey of Low-Resource Machine Translation
https://aclanthology.org/2022.cl-3.6.pdf

- In neural machine translation, what does transfer learning transfer? (studied the effects of transferring embeddings in transfer learning settings)
https://aclanthology.org/2020.acl-main.688/

