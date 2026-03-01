# LingML

## Directory Structure

- **config** - configuration files for different datasets and LLM models
- **data_classes** - Python classes to handle different datasets, and make them suitable for training
- **datasets** - raw datasets in csv format
    - **aaai-constraint-covid** - original dataset by [Patwa et al., 2020](https://arxiv.org/abs/2011.03327/)
    - **aaai-constraint-covid-appended** - original dataset with appended linguistic features retrieved using [LIWC-22](https://www.liwc.app/)
    - **aaai-constraint-covid-cleaned** - dataset constructed by eliminating records identified by [Bee et al., 2023](https://arxiv.org/abs/2310.04237), as being neither true nor false in the context of COVID-19
    - **aaai-constraint-covid-cleaned-appended** - cleaned dataset with appended linguistic features
- **model_classes** - Python classes to handle 11 transformer-based LLMs
    - all models need special implementation for incorporating language features
    - number of output heads needed to be changed from 3 to 2 for Twitter-RoEBRTa
- **results** - results of the different runs <br>
    - **directory structure** - \<dataset> -> \<model> -> \<run-date> -> training logs and <data-split_results>
- **utils** - utility functions for running the transformer experiments

- **main.py** - main file for running the transformer experiments


## Setup

```bash
conda create --name <env-name> --file requirements.txt python=3.8
conda activate <env-name>
```

## Execution

To run the transformer experiments, execute
```bash
python3 -B main.py \
    --dataset <dataset> \
    --model <model>
```

where dataset can be one of
- **aaai-constraint-covid**
- **aaai-constraint-covid-appended**
- **aaai-constraint-covid-cleaned**
- **aaai-constraint-covid-cleaned-appended**

and model can be one of 
- **albert-base-v2** - Base version of ALBERT model with a randomly initialized sequence classification head. See [HF model card](https://huggingface.co/albert-base-v2).

- **bertweet-covid-19-base-uncased** - Base version of BERTweet model, a RoBERTa model pre-trained on ~850M tweets, ~5M of which were COVID-19 related. See [HF model card](https://huggingface.co/vinai/bertweet-covid19-base-uncased).
- **covid-twitter-bert-v2** - CT-BERT Model, which is a large BERT model pre-trained on ~97M COVID-related tweets. See [HF model card](https://huggingface.co/digitalepidemiologylab/covid-twitter-bert-v2).


You can also override default configurations using the command line. For example,
```bash
python3 -B main.py \
    --dataset <dataset> \
    --model <model> \
    ADD_NEW_TOKENS True \
    DATASET.BATCH_SIZE 16 \
    DATASET.args.root <dataset-root> \
    MODEL.MAX_LENGTH 200
```

For inference, execute
```bash
python3 -B inference.py \
    --dataset <dataset> \
    --model <model> \
    --weights <path-to-weights>
```

For example,
```bash
python3 -B inference.py \
    --dataset aaai-constraint-covid \
    --model covid-twitter-bert-v2 \
    --weights "./results/aaai-constraint-covid/CT-BERT/2025-10-19-00-28-08/ckpt5350.pth"
```

Note: Make sure to set the device index to <i>None</i> if you do not wish to use the GPU, i.e.
```bash
python3 -B main.py \
    --dataset <dataset> \
    --model <model> \
    DEVICE_INDEX None
```

