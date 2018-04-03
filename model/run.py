import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys
import re
import pickle
from datetime import datetime

from music21 import instrument

import functions as fxn

# midi_files = '../../data/Music/Tadpole/**/*.MID'
# midi_files = '../audio_output/Dance/2_10A_201804011556-200-lstm_midi.mid' # for testing
midi_files = '../../data/Music/MidiWorld/Classical/*.mid'
# midi_files = '../../data/Music/FinalFantasy/*.mid'

timestamp = re.sub(r'[-: ]','',str(datetime.now()).split('.')[0])[:-2]
output_name = midi_files.split('/')[-2]
total_epochs = 50
batch_size = 512 # 128 for local; 512 for AWS
sequence_length = 200 # the LSTM RNN will consider this many notes
notes_generated = 500
temperature = 1.0
offset_adj = 0.5

output_tag = 'output/{}-{}-'.format(timestamp, output_name)
# sound = instrument.Bagpipes()


def full_execution(midi_files, output_tag, total_epochs, batch_size, sequence_length, temperature, offset_adj):
    # epoch_stops = 1
    # epoch_count = 0
    weight_file = None

    note_file = fxn.convert_midis_to_notes(midi_files, output_tag)

    epochs = total_epochs
    with open(note_file, 'rb') as filepath:
        notes = pickle.load(filepath)
    network_input, network_output, n_patterns, n_vocab, pitchnames = fxn.prepare_sequences(notes, sequence_length)
    network_input_r, network_output_r = fxn.reshape_for_training(network_input, network_output,sequence_length)

    # while epoch_count <= total_epochs:
    #     epochs = epoch_stops
    #
    model = fxn.create_network(network_input_r, n_vocab, weight_file)
    model, weight_file, history = fxn.train_model(model, network_input_r, network_output_r, epochs, batch_size, output_tag, sequence_length)
    normalized_input = fxn.reshape_for_creation(network_input, n_patterns, sequence_length, n_vocab)
    model = fxn.create_network(normalized_input, n_vocab, weight_file)
    prediction_output= fxn.generate_notes(model, network_input, pitchnames,sequence_length, notes_generated, n_vocab, temperature)
    output_notes = fxn.create_midi(prediction_output, output_tag, sequence_length, offset_adj)
        # epoch_count += epoch_stops
    return output_notes, history, weight_file


terminal_output = output_tag + 'terminal.log'
sys.stdout = fxn.Logger(terminal_output)
print("Terminal output being saved at {}".format(terminal_output))
full_execution(midi_files, output_tag, total_epochs, batch_size, sequence_length, temperature, offset_adj)
print("Run Complete. Terminal log saved at {}".format(terminal_output))
