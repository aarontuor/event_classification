# Event Classification #

This repo contains the python scripts and modules, and data for pre-processing text, training and using 
a recurrent neural network (RNN) active event classifier. 

## Dependencies ##


All code in the repo was written in Python 2 using Tensorflow. The GPU version of Tensorflow was used
but the code is also compatible with CPU Tensorflow. The complete Python environment is recorded in
``requirements.txt``. The operating system used in code development was Linux Mint 18.

## Files ##

* ``requirements.txt``: Python environment description.
* ``make_lstm_classifier.py``: Script which trains the RNN classifier given pre-processed data. 
* ``lstm_classifier.py``: Contains class which loads pre-trained model for predicitons (testing code included in main).
* ``preprocess.py``: Contains class which preprocesses individual lines of text (testing code included in main).
* ``patterns.py``: Regular expression definitions.
* ``tf_ops.py``: Tensorflow function definitions.
* ``graph_training_utils.py``: Boiler plate code for training models in tensorflow packaged as classes and functions.
* ``util.py``: Grab bag of helper functions.
* ``event_desc_dict.txt``: File with line for every word in the data set to be included in the dictionary. 
* ``input_data/``
    - ``make_datasplit.py``: Script for splitting the data into 80/10/10, train/dev/test split.
    - ``patterns.py``: Regular expression definitions for replacing URL's, email addresses, etc...
    - ``preprocess_dataset.py``: Script for cleaning up text for input to the model. 
    - ``text_to_int.py``: Script for translating normalized text into sequences of integers for model input.
    - ``raw/`` 
        + ``desc.txt``: Plain text file with 1 line per description of article in the data set.
        + ``labels.txt``: Plain text file with 1 line per label. Label is "1" for active event and "0" otherwise
        + ``title.txt``: Plain text file with 1 line per title of an article in the data set.
    - ``normalized/``: Contains the output after running ``preprocess_dataset.py`` on the contents of the `raw` folder.
        + ``desc_processed.txt``: Plain text file with 1 line per normalized description of article in the data set.
        + ``label_processed.txt``: Plain text file with 1 line per label. Label is "1" for active event and "0" otherwise
        + ``title_processed.txt``: Plain text file with 1 line per normalized title of an article in the data set.
    - ``split/``: Contains the output after running ``make_datasplit.py`` on the contents of the ``normalized`` folder. 
    - ``numpy/``: Contains the output after running ``text_to_int.py`` on the contents of the``split`` folder.
        + ``*_desc.npy``: A matrix (NUM_ARTICLES X 68) where:
            - the first column is a unique ID for each article, 
            - the second is a column of 0's, 
            - the third the class label, 
            - the fourth the number of words in the text and
            - the remaining columns are sequences of integers representing the description text.
        + ``*_title.npy``: A matrix (NUM_ARTICLES X 40) of the same format as ``*_desc.npy``.
        + ``desc_event_biovects.npy``: A matrix (VOCABSIZE X 256)of pre-trained GloVe embeddings for words in the vocabulary. 
        Order of vectors corresponds to order of words in ``event_desc_dict.txt``.
    - ``saved_model``: Checkpoint files for loading the trained model to make predictions.
## Usage ##

### Preparing the Data ###
1. Make a copy of this repo. 
2. In the repo copy replace the files in the ``raw`` folder with examples from the new data set to train on.
3. From the command line navigate to the ``input_data`` directory, then:

    ```bash 
    $ python preprocess_dataset.py
    $ python make_datasplit.py
    $ python text_to_int.py
    ```

### Training ###

    

To train the model run make_lstm_classifier.py from the directory it is located in. 
```bash
$ python make_lstm_classifier.py

```

This will take several hours to days to complete training depending on the size of the data set 
and the hardware on your system. 
For the 31,000 article dataset training on 3 epochs completed in about 15 hours using a single GPU. 
Training on the CPU will take significantly longer. 

To confirm that the model has trained sufficiently run:

```bash
$ python lstm_classifier.py
```
    
The accuracy should be in the high 90 percent range. 
    

## Performance ##
Timing for 3000 articles sequentially pre-processed and predicted: 137 seconds (CPU), 70 seconds (GPU)
