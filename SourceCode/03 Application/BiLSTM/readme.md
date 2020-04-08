#### BiLSTM.ipynb

This file records our training process to get the balanced or weighted BiLSTM models.

You could also run this code to train the sentiment classification model.

1. Put train.csv and test.csv under this directory. Those two dataset can be generate by file make_dataset.py.
2. Run this code in colab with GPU backend.

#### make_dataset.py

Generate train.csv and test.csv. These two datasets are used in training BiLSTM.

Change the original sentence label.  Remove punctuation marks.

- Postive [4,5] --- 2  
- Negative [1,2]---0 
- Neutral 3---1

1. Install pandas
2. Put 'CellPhoneReview.json' under this directory.

#### evaluation.ipynb

Evaluate the model on the test dataset. Calculate the f1-score, precision, recall, accuracy. 

Put a sentence to the model and get sentiment classification result as well as probability.

1. Put train.csv and test.csv under this directory. Those two dataset can be generate by file make_dataset.py.
2. Load models 
   - Download pretrained models or train new models
   - [best_model_hidden_128_eval_weighted.pt](https://drive.google.com/open?id=1-9WpnPnQnABermTc-MbH_MxhaQyg1tZh)
   - [best_model_hidden_128_eval.pt](https://drive.google.com/open?id=1-3kUIndJRKhNhzhBwKtCJ_sFNCdgNAIS)
3. Run this code in colab with GPU backend.

#### classify.py

Classify a sentence by using model.

Two parameters:

- -n : model filename 
- -s : input sentence

Run like this  **python classify.py -s "Five stars"**

Output: Sentiment Label , Probability

Namespace(name='best_model_hidden_128_eval_weighted.pt', sentence='Five stars')

Length of Text Vocabulary: 188707

Vector size of Text Vocabulary:  torch.Size([188707, 300]) 

Label Length: 3 

Five stars 

Sentiment: Positive 

[0.4331267774105072, 0.16560469567775726, 0.4012685716152191]

1. Put train.csv, test.csv, BiLSTM.py under this directory. Those two dataset can be generate by file make_dataset.py.
2. Load models 
   - Download pretrained models or train new models
   - [best_model_hidden_128_eval_weighted.pt](https://drive.google.com/open?id=1-9WpnPnQnABermTc-MbH_MxhaQyg1tZh)
   - [best_model_hidden_128_eval.pt](https://drive.google.com/open?id=1-3kUIndJRKhNhzhBwKtCJ_sFNCdgNAIS)
3. Run this code in colab with GPU backend.

#### BiLSTM.py

Model definition.