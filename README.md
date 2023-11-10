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

- Augmenting mBART with Back Translation and Transfer Learning to survey benefits in low resource NMT. Demonstrated benefits on Ne-En with Back Translation (Could be one of our baselines if our implemented model is better than our first choice of baseline)
https://arxiv.org/pdf/2304.07869.pdf

- Incorporating BERT into NMT (Demonstrated improvements in high resource languages like English, German, and French. Also did not use heavier models like mBART. Used two techniques - Initialized embeddings and Used the output embeddings as context-aware embeddings for NMT)
https://openreview.net/attachment?id=Hyl7ygStwB&name=original_pdf

- Recycling a pretrained BERT for NMT https://aclanthology.org/D19-5603.pdf

- Enhancing use of BERT information in neural machine translation https://d197for5662m48.cloudfront.net/documents/publicationstatus/113656/preprint_pdf/75139e5fc2b7799ed8081cbeb0405849.pdf

- Towards Making the Most of BERT in Neural Machine Translation (Catastrophic forgetting for resource rich tasks) https://scholar.google.com/scholar?q=Towards%20Making%20the%20Most%20of%20BERT%20in%20Neural%20Machine%20Translation

- BERT-JAM: Maximizing the utilization of BERT for neural machine translation (Issues with only using the last layer) https://arxiv.org/pdf/2011.04266.pdf (No code available for this paper)

- Adaptive Adapters: An Efficient Way to Incorporate BERT Into Neural Machine Translation
http://staff.ustc.edu.cn/~linlixu/papers/tasl21.pdf

- Survey of Low-Resource Machine Translation
https://aclanthology.org/2022.cl-3.6.pdf

- In neural machine translation, what does transfer learning transfer? (studied the effects of transferring embeddings in transfer learning settings)
https://aclanthology.org/2020.acl-main.688/

