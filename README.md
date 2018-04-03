# AI Jukebox
The AI Jukebox takes as input a collection of midi files and outputs AI generated music.  The underlying model is a bidirectional LSTM recurrent neural network which maps the latent space of the collection of files and then samples from this underlying structure.  

**Music Samples** 
See music samples on [SoundCloud.](https://soundcloud.com/cipher813)

The AI Jukebox works in two phases: Training and Evaluation.  

The model takes in a set of midi music files and generates a unique note/chord pattern in midi output.  This is processed in two phases: (1) Midi input files to trained weights, and (2) Trained weights to midi generated file.  

Midis -> weights -> midi

The model consists of two files:

`functions.py` contains the internal structure of the model for both training and evaluation stages.  
`run.py` executes the model from beginning of training to end of evaluation.  

The model default is to generate a weight file at the end of every epoch.  Once you have a weight file and the input notes file generated at the beginning of training, you can convert these files into a midi of generated music.  

### Training

The training phase is quite time consuming; this is where the model maps the latent space within the collection of music.  A weight file is generated at the end of this phase.  

### Evaluation

Utilizing the weight file generated from the training phase, the model will then "predict" a sequence of notes which represents a sample from the internal structure mapped.  A midi file is generated as output.  

To run the model from beginning of training phase to end of evaluation phase, simply navigate to the model folder and then run the program by typing `python run.py`

If you have already created weight files from the training phase (partially trained weights will work as well), you can generate a midi file from the weight file.  To do so, simple open the weights_to_midi.ipynb jupyter notebook and input the weight filepath and input notes filepath accordingly.  Input notes file will be created when the model is first trained.  


### Resources
Nayebi, Aran. ["GRUV: Algorithmic Music Generation using Recurrent Neural Networks."](https://www.arxiv.org). Stanford University. 2015.  

Skúli, Sigurður.  ["How to Generate Music using a LSTM Neural Network in Keras."](https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5). December 7, 2017.

Brownlee, Jason. ["Stacked LSTM Networks."](https://machinelearningmastery.com/stacked-long-short-term-memory-networks/). August 18, 2017.

Brownlee, Jason. ["Understand the Difference Between Return Sequences and Return States for LSTMs in Keras."](https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/). October 24, 2017.  
