

#  Simple and effective architectures for Nested NER
======================================================================

## Install

1. Create an enviroment: `python -m venv venv` and activate it.
2. Run `pip install -r requirements.txt` to install all dependencies
3. In case you use a GPU, then install this PyTorch version: `pip3 install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html`


## Files

Sequence Multilabeling architecture files:


Neural Layered architecture files:

1. After having obtained the files for the WL Corpus, run the command: `python neural_layered_data.py`,  to obtain Neural Layered files in neural_layered_files. Files will be located in the neural_layered_files folder.

## Embeddings

Put the `cwlce.vec` embeddings (it can be downloaded from here: https://zenodo.org/record/3924799).

The BERT and Flair contextual embeddings are generated using this code: https://github.com/zalandoresearch/flair. 

The selection of embeddings to be used can be modified in the `params.json` file.

## SML Training.

Training parameters can be changed in `params.json` file

Run the script `main.py`. The results will be will be printed to console, the models will be saved in models folder.

The models will be stored in the models folder.

## Neural Layered experiments

We use the original repository of the paper: https://github.com/meizhiju/layered-bilstm-crf using files obtained in this repository. 