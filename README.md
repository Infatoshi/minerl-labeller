# minerl-labeller

Inspired by https://github.com/openai/Video-Pre-Training

This repo is the first stage to pretrain a minecraft vision model on a bunch of YouTube videos of gameplay (getting the model to mimic human gameplay).
The first stage is to create a labeller model on some labelled data a human has created. 
Once the labeller has sufficient performance, you can begin the second stage (will talk about this in the next repo).

## Steps:
`git clone https://github.com/Infatoshi/minerl-labeller`
`cd minerl-labeller`
`python3 -m venv myvenv`
Either `source myvenv/bin/activate` or `myvenv/Scripts/activate` depending on your OS
Install the JavaSDK and minerl: https://minerl.readthedocs.io/en/latest/tutorials/index.html
`pip install gym cv2 numpy pygame torch`
Install PyTorch for CUDA: https://pytorch.org/get-started/locally/
Create training data directory: 
`mkdir labeller-training`
`cd labeller-training`
`mkdir action`
`mkdir video`
`cd ..`
Collect your data:
`python human-data-maker.py`
Train model:
`python architecturev3.py`
