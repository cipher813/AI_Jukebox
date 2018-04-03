# LSTM Music Generator
Consists of the following files, each saved under their version number in the models folder:

**Run files**
The model takes in a set of midi music files and generates a unique note/chord pattern in midi output.  This is processed in two phases: (1) Midi input files to trained weights, and (2) Trained weights to midi generated file.  

Midis -> weights -> midi

`midis_to_weights.py` takes in set of midi files and trains weights.
`weights_to_midi.py` takes the trained weights and generated a unique midi file.

**Support files:**
`processing.py` contains functions that process notes/chords from midi to embedded numeric format.  
`neural_network.py` contains the neural network which trains the weights and generates the midi output by "predicting" notes/chords.  
`generate.py` contains functions which create the midi output.
`utils.py` contains a logging function for process tracking.  


**To run:**

Navigate to the model files in within the model/<version number> filepath.

First, type `python midis_to_weights.py` to train by inputting midi files and outputting a trained set of weights.

Once you have the trained weights file, edit `weights_to_midi.py` with filepaths to the weights file and notes file produced as output to the training phase.  Then run this process in terminal by typing `python weights_to_midi.py`

This will then output the generated midi file.  


**Resources:**
Nayebi, Aran. ["GRUV: Algorithmic Music Generation using Recurrent Neural Networks."](https://www.arxiv.org). Stanford University. 2015.  

Skúli, Sigurður.  ["How to Generate Music using a LSTM Neural Network in Keras."](https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5). December 7, 2017.

Brownlee, Jason. ["Stacked LSTM Networks."](https://machinelearningmastery.com/stacked-long-short-term-memory-networks/). August 18, 2017.

Brownlee, Jason. ["Understand the Difference Between Return Sequences and Return States for LSTMs in Keras."](https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/). October 24, 2017.  
