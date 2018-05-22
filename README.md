# Sentiment-Classifier
This program detects the emotions/context associated with an input sentence, and prints out emojis to represent them! Trained on an RNN model.

The dataset is quite small, but feel free to expand! Words outside of the training set are not included in the embedding dictionary that's created, and thus will be erased- be cautious in using uncommon words!

E.g. "John wants food" will be read by the program as "wants food" 

IMPORTANT NOTE: The 100D pre-trained GloVe embedding is not included in the repository due to size limitations.

Download it from this link:
https://nlp.stanford.edu/projects/glove/ (400K vocab)

Once downloaded, move the 100d embedding into the directory for this classifier.
