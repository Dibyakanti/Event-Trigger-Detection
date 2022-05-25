# Event-Trigger-Detection
This is the code used for Event Trigger Detection in short stories and novels.

## Dataset :
[LitBank](https://github.com/dbamman/litbank) is the primary dataset used here but this code is compatible with any data as long the annotaion files and text files are stored in the way mentioned below.

## Preprocessing :
In `LitBank` the dataset consists of two files obtained from `brat`. The `litevent_data_gen.ipynb` contains the code necessary to convert the `brat` outputs into tsv files with a label for each word.

## Sentence classifier :
The `litevent_sentence_classifier.ipynb` contains the code necessary for classifying if each sentence has an event or not.

## Word classifier :
The `litevent_word_classifier.ipynb` contains the code necessary for classifying if each word is an event or not.
