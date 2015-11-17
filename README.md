# NaiveBaes

## Dependencies
Just install Anaconda man.

## Usage
### usage: DigitNaiveBayes.py [-h] [-k K] [-f NUM_FEATURES] runmode  

positional arguments:  
  runmode               Determine which dataset to use: digits or faces  

optional arguments:  
  -h, --help            show this help message and exit  
  -k K                  Smoothing factor (1 - 50)  
  -f NUM_FEATURES, --num_features NUM_FEATURES (2 or 3)  
  
  
### usage: TextNaiveBayes.py [-h] document_type runmode k_value  
  
positional arguments:  
  document_type  Choose a type: spam_detection, movie_reviews, 8_newsgroups  
  runmode        Choose a runmode: multinomial, bernoulli  
  k_value        Choose a k value for Laplacian smoothing  
  
optional arguments:  
  -h, --help     show this help message and exit  
