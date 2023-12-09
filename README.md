# Source Language Expert-Driven Low-Resource NMT

## Introduction

We aim to leverage NepBERTa as a pre-trained source language expert for improving the translations of Nepali to English. We merge the NepBERTa model with the model of mBART by fusing their embeddings together. Details about the architecture can be found in the github repo.

## Training the model

### Getting raw data for Nepali and English

We use the publically available corpora corresponding to GNOME/KDE/Ubuntu localization files (495K sentences, 2M tokens). These can be downloaded from https://opus.nlpl.eu. All the train, dev, and test data, along with the relevant scripts, are present in the **dataset** folder. Note that the language code for English is `en_XX` and the language code for Nepali is `ne_NP`.

We download and combine the raw training data from these sources by using the script _combineTrain.py_. Upon analysing this dataset, we observed that there were certain noisy training pairs and repetitions in this overall training set. To get combat any possible training detorioration due to this unclean data, we perform some cleaning steps which are included in _cleanTrain.py_

We use the FLORES-200 evaluation dataset to get the dev and the test set for Nepali and English. This data can be downloaded here: https://github.com/facebookresearch/flores/tree/main/previous_releases/floresv1.

Due to size constraints, we could not upload the training data corresponding to the other languages of Sinhala, Pashto, and Khmer that we analysed in our paper.

### Reproducing the results

#### MBART baseline:

- The training scripts that were used to get the results for the mBART model can be found in the **mBART_Baseline** directory.
- Run `mbart50.ipynb`
- Set Dir path to point to location of **dataset** folder
- Set `fine_tune`` to True or False depending on if model is to be finetuned
- Run all cells to get results

#### mT5 baseline:

- The training scripts that were used to get the results for the mBART model can be found in the **MT5_Baseline** directory.
- Run `mt5.ipynb`

#### Our system: NepBERTa + mBART

- The scripts corresponding to the implementation, data-loading, fine-tuning, and testing of our model of NepBERTa + mBART could be found in the **mBART_NepBERTa** directory.
- To fine-tune the system such that all the parameters are getting trained (ref: _Addition & Train-All-Params_), run `python finetune.py -f` to use the default settings of LR=1e-5 and batch size of 8, with parallel utilization of GPUs. You can tweak around different hyper-parameters by passing in different values of flags as seen in the fine-tuning script.
- To fine-tune the system such that only the parameters corresponding to the MLP adapter are getting trained (ref: _Addition & Train-Only-Adapter_), checkout to the branch **vedant/frozen-params** and run the same fine-tuning script as described above.
- If you wish to monitor the performance of the system while it's training, you could setup an account on wandb (https://wandb.ai), and pass the flag `-w` in the fine-tuning script. There you can monitor the BLEU scores, Training loss, Dev loss, GPU usage, and a lot of other necessary stuff.
- To use the _Pretrained mBART_ for the experiments, use `mBart_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50", output_hidden_states=True)` in the scripts in the current directory (i.e. **mBART_NepBERTa**)
- To use the _Finetuned mBART_ for the experiments, use `mBart_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", output_hidden_states=True)` in the scripts in the current directory (i.e. **mBART_NepBERTa**)
- To test the performance of a particular saved checkpoint, simply run `test.py` after having modified the path to the saved check-point appropriately.

#### Our system (not experimented with): NepBERTa + mT5

- We combine the pre-trained models of NepBERTa and mT5 using an architecture similar to the one used for fusing NepBERTa and mBART together. Due to lack of available compute resources, we could not perform the similar experimentations for this system. Hence we have not included this as one of our contributions.
- The scripts corresponding to the implementation, data-loading, fine-tuning, and testing of this system can be found in directory **T5_NepBERTa**.
- To fine-tune the system, run `python finetune.py -f` to use the default settings of LR=1e-5 and batch size of 8, with parallel utilization of GPUs. You can tweak around different hyper-parameters by passing in different values of flags as seen in the fine-tuning script.
