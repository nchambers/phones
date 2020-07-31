# Phone Number Extraction
Neural models for phone number extraction from adversarial text.

This code contains the experimental models described in the paper: "Character-Based Models for Adversarial Phone Number Extraction: Preventing Human Sex Trafficking" from Nathanael Chambers, Timothy Forman, Catherine Griswold, Kevin Lu, and Stephen Steckler.

The biLSTM, CRF, and CNN-input models are included in the code.



Setup
-------------

You need to also download the unicodeviz repository:
https://github.com/nchambers/unicodeviz

Create an environment variable *UNICODEVIZ* that points to its imgs/ subdirectory

The code was developed on TensorFlow 1.14.0 and Keras 2.2.4. It may not work on later versions of either one.


How to Run
-------------

(baseline) Training the base biLSTM without a CRF:

      python3 phoneRNN.py -con -att train rnn.train

(good model) Training the CRF:

      python3 phoneCRF.py train crf.train

Training the CRF with CNN visual characters:

      export UNICODEVIZ='path/to/repo/imgs/'
      python3 phoneCRF.py -cnn train crf.train

Testing a trained CRF model (modelname is output from the above training):

      python3 phoneCRF.py test <modelname> rnn.test
