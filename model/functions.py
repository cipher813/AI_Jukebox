import sys
import glob
import pickle
import numpy as np

from music21 import converter, note, chord, stream, instrument

from keras import optimizers
from keras.models import Sequential, load_model
from keras.layers import Bidirectional, Dense, Dropout, LSTM, Activation
from keras.callbacks import ModelCheckpoint, TensorBoard, History, Callback
from keras.utils import np_utils

# from keras.layers.core import K


# PROCESSING


def convert_midis_to_notes(midi_files, output_tag):
    # convert midi file dataset to notes
    notes = [] # list of notes and chords
    note_count = 0

    print("\n**Loading Midi files**")
    for file in glob.glob(midi_files): # loading midi filepaths
        print(file)
        try:
            midi = converter.parse(file) # midi type music21.stream.Score
            parts = instrument.partitionByInstrument(midi)

            if parts:
                notes_to_parse = parts.parts[0].recurse()
            else:
                notes_to_parse = midi.flat.notes
            # notes_to_parse type music21.stream.iterator.RecursiveIterator
            for e in notes_to_parse:
                if isinstance(e, note.Note):
                    notes.append(str(e.pitch))
                elif isinstance(e, chord.Chord):
                    to_append = '.'.join(str(n) for n in e.normalOrder)
                    notes.append(to_append)
            note_count +=1
        except Exception as e:
            print(e)
            pass
    assert note_count > 0
    n_vocab = len(set(notes))
    print("Loaded {} midi files {} notes and {} unique notes".format(note_count, len(notes), n_vocab))

    note_file = output_tag + 'input_notes'
    with open(note_file, 'wb') as f:
        pickle.dump(notes, f)
    print("Input notes/chords stored as {} then pickled at {}".format(type(notes), note_file))
    print("First 20 notes/chords: {}".format(notes[:20]))
    return note_file


def prepare_sequences(notes, sequence_length):
    print("\n**Preparing sequences for training**")
    pitchnames = sorted(set(i for i in notes)) # list of unique chords and notes
    n_vocab = len(pitchnames)
    print("Pitchnames (unique notes/chords from 'notes') at length {}: {}".format(len(pitchnames),pitchnames))
    # enumerate pitchnames into dictionary embedding
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    print("Note to integer embedding created at length {}".format(len(note_to_int)))

    network_input = []
    network_output = []

    # i equals total notes less declared sequence length of LSTM (ie 5000 - 100)
    # sequence input for each i is list of notes i to end of sequence length (ie 0-100 for i = 0)
    # sequence output for each i is single note at i + sequence length (ie 100 for i = 0)
    for i in range(0, len(notes) - sequence_length,1):
        sequence_in = notes[i:i + sequence_length] # 100
        sequence_out = notes[i + sequence_length] # 1

        # enumerate notes and chord sequences with note_to_int enumerated encoding
        # network input/output is a list of encoded notes and chords based on note_to_int encoding
        # if 100 unique notes/chords, the encoding will be between 0-100
        input_add = [note_to_int[char] for char in sequence_in]
        network_input.append(input_add) # sequence length
        output_add = note_to_int[sequence_out]
        network_output.append(output_add) # single note

    print("Network input and output created with (pre-transform) lengths {} and {}".format(len(network_input),len(network_output)))
    # print("Network input and output first list items: {} and {}".format(network_input[0],network_output[0]))
    # print("Network input list item length: {}".format(len(network_input[0])))
    n_patterns = len(network_input) # notes less sequence length
    print("Lengths. N Vocab: {} N Patterns: {} Pitchnames: {}".format(n_vocab,n_patterns, len(pitchnames)))
    return network_input, network_output, n_patterns, n_vocab, pitchnames


def reshape_for_training(network_input, network_output,sequence_length):
    print("\n**Reshaping for training**")
    n_patterns = len(network_input)
    # convert network input/output from lists to numpy arrays
    # reshape input to (notes less sequence length, sequence length)
    # reshape output to (notes less sequence length, unique notes/chords)
    network_input_r = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_output_r = np_utils.to_categorical(network_output)

    print("Reshaping network input to (notes - sequence length, sequence length) {}".format(network_input_r.shape))
    print("Reshaping network output to (notes - sequence length, unique notes) {}".format(network_output_r.shape))
    return network_input_r, network_output_r


def reshape_for_creation(network_input, n_patterns, sequence_length, n_vocab):
    print("\n**Preparing sequences for output**")
    n_patterns = len(network_input)
    # the network input variables below are unshaped (pre-reshaped)
    normalized_input = np.reshape(network_input, (n_patterns, sequence_length,1)) / float(n_vocab)
    print("Network Input of length {} is reshaped to normalized input of {}".format(len(network_input),normalized_input.shape))
    return normalized_input


# NEURAL NETWORK


first_layer = 512
drop = 0.5

# K.set_learning_phase(0)

def create_network(network_input, n_vocab, weight_file):
    print("\n**LSTM model initializing**")
    # this is a complete model file

    # network input shape (notes - sequence length, sequence_length, 1)
    timesteps = network_input.shape[1] # sequence length
    data_dim = network_input.shape[2] # 1

    print("Input nodes: {} Dropout: {}".format(first_layer, drop))
    print("Input shape (timesteps, data_dim): ({},{})".format(timesteps, data_dim))
    # for LSTM models, return_sequences sb True for all but the last LSTM layer
    # this will input the full sequence rather than a single value
    model = Sequential()
    model.add(Bidirectional(LSTM(first_layer, recurrent_dropout=drop), input_shape=(timesteps, data_dim)))
    model.add(Dense(first_layer))
    model.add(Dropout(drop))
    model.add(Dense(n_vocab)) # based on number of unique notes
    model.add(Dropout(drop))
    model.add(Activation('softmax'))

    try:
        model.load_weights(weight_file)
        print("Weights file loaded")
    except Exception as e:
        print(e)
        print("No weights loaded")
        pass

    rms = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay = 0.0)

    model.compile(loss='categorical_crossentropy',optimizer=rms)

    return model


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('val_loss'))


def train_model(model, network_input_r, network_output_r, epochs, batch_size, output_tag, sequence_length):
    # saves model after each epoch
    # check_stats = '{epoch:02d}-{loss:.4f}-{val_loss:.4f}-'
    # weight_file = output_tag + check_stats + 'weights.hdf5'
    base_tag = output_tag + 'weight-'
    epoch_metrics = '{epoch:02d}-{loss:.4f}-{val_loss:.4f}'
    end_tag = '.hdf5'
    weight_checkpoint = base_tag + epoch_metrics + end_tag
    checkpoint = ModelCheckpoint(weight_checkpoint,
                                    monitor='val_loss',
                                    verbose=1,
                                    save_best_only=False,
                                    mode='min') #, period=1)

    # https://stackoverflow.com/questions/42112260/how-do-i-use-the-tensorboard-callback-of-keras?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    tensorboard = TensorBoard(log_dir='log', histogram_freq=1, write_graph=True, write_images=True)

    history = LossHistory()

    callbacks_list = [checkpoint, tensorboard, history]

    print("Fitting Model. \nNetwork Input Shape: {} Network Output Shape: {}".format(network_input_r.shape,network_output_r.shape))
    print("Epochs: {} Batch Size: {}".format(epochs, batch_size))

    # save history
    # https://stackoverflow.com/questions/41061457/keras-how-to-save-the-training-history?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    history = model.fit(network_input_r, network_output_r, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list, validation_split=0.2)
    history_filepath = output_tag + 'history.pkl'
    with open(history_filepath, 'wb') as f:
        pickle.dump(history.history,f)
    print("History: {}".format(history.history))
    print("History saved at {}".format(history_filepath))

    # saves model upon training completion
    weight_file = output_tag + '-last_weights.hdf5'
    model.save_weights(weight_file)
    print("TRAINING complete - weights saved at: {}".format(weight_file))
    return model, weight_file, history




# CREATE MIDI


# sample function from Keras Nietsche example
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_notes(model, network_input, pitchnames,sequence_length, notes_generated, n_vocab, temperature):
    # diversity_list = [0.2,0.5,1.0,1.2]
    print("\n**Generating notes**")
    # convert integers back to notes/chords
    print("Length Pitchnames: {}".format(len(pitchnames)))
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    # note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    print("Integer to note conversion at length: {}".format(len(int_to_note)))

    # randomly instantiate with single number from 0 to length of network input
    # network_input = network_input[1:]
    print("Network input length: {}".format(len(network_input)))
    # try:
    start = np.random.randint(0,len(network_input)-1)
    # except Exception as e:
    #     print(e)
    #     start = sequence_length
    #     pass
    # for diversity in [0.2, 0.5, 1.0,1.2]:

    # generated = ''
    # pattern = network_input[start: start + 100]
    # generated += str(pattern)

    pattern = network_input[start]
    # #
    prediction_output = []
    # print("Pattern begins with length {} and type {}".format(len(pattern),type(pattern)))
    # print("Pattern: {}".format(pattern))
    # # for each note in notes generated declared as hyperparameter above (ie 500)
    for note_index in range(notes_generated):
        # x_pred = np.zeros((1,100,n_vocab))
        # for t, char in enumerate(pattern):
        #     print("T: {} Char: {}".format(t,char))
        #     x_pred[0,t,note_to_int[char]] = 1.
        # pattern = sample(pattern, diversity)
        prediction_input = np.reshape(pattern, (1,len(pattern),1)) / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)[0]
        # diversity = diversity_list[np.random.randint(0,4)]
        index = sample(prediction,temperature)
        # index = np.argmax(prediction)
        result = int_to_note[index]

        # prediction_output += result
        prediction_output.append(result)
        #
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
        # print("Types. Note index: {} prediction_input: {} prediction: {} index: {} result: {}".format(type(note_index),type(prediction_input),type(prediction),type(index),type(result)))

    print("Pattern ends with length {} and type {}".format(len(pattern),type(pattern)))
    print("Generated Note Length: {}\nFirst 100: {}".format(len(prediction_output), prediction_output[:100]))
    return prediction_output


def create_midi(prediction_output, output_tag, sequence_length,offset_adj):
    print("\n**Creating midi**")
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        # prepares chords (if) and notes (else)
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Flute()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Flute()

            output_notes.append(new_note)
        offset += offset_adj #0.5

    print("Generating {} notes stored as {}".format(len(output_notes),type(output_notes)))
    midi_stream = stream.Stream(output_notes)
    midi_file = output_tag + 'lstm_midi.mid'
    midi_stream.write('midi',fp=midi_file)
    print("Midi saved at: {}".format(midi_file))

    output_notes_file = output_tag + 'output_notes'
    with open(output_notes_file, 'wb') as f:
        pickle.dump(output_notes, f)
    print("Output notes/chords stored as {} then pickled at {}".format(type(output_notes), output_notes_file))
    return output_notes, midi_file


class Logger(object):

    def __init__(self, terminal_output):
        self.terminal = sys.stdout
        self.log = open(terminal_output, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass
