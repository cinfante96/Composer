import glob
import pickle
import argparse
import os
import uuid
import numpy as np
from random import shuffle
from music21 import converter, instrument, note, chord, stream
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import optimizers

def createDir(directory):
    if not os.path.exists(directory):
        print("{} doesn't exist. Creating.".format(directory))
        os.makedirs(directory)

def readMids(dataset):
    '''
    Reads every midi in midi_files/dataset directory.
    Saves a list of the notes in a file under notes/ directory.
    '''
    notes = []

    for file in glob.glob("midi/{}/*.mid".format(dataset)):
        midi = converter.parse(file)

        print("Reading from %s" % file)

        notes_to_parse = None

        try:
            score = instrument.partitionByInstrument(midi)
            notes_to_parse = score.parts[0].recurse() 
        except:
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    shuffle(notes)
    createDir("notes")
    fname = 'notes/{}'.format(dataset)
    if os.path.exists(fname):
        print("File already exists.")
        fname += str(uuid.uuid4())
    with open(fname, 'wb') as filepath:
        pickle.dump(notes, filepath)

    return fname

def encode(args):
    fname = readMids(args.dataset)
    print("Dataset encoded and saved in ",fname)

def getXY(notes, vocab, seq_len):
    '''
    Creates a dictionary and uses it to map integers to notes.
    Then, uses the dictionary and creates normalized input and output for the network.
    '''

    pitchstrings = sorted(set(item for item in notes))

    note2int = dict((note, number) for number, note in enumerate(pitchstrings))

    input = []
    output = []
    for i in range(0, len(notes) - seq_len, 1):
        sequence_in = notes[i:i + seq_len]
        sequence_out = notes[i + seq_len]
        input.append([note2int[char] for char in sequence_in])
        output.append(note2int[sequence_out])

    n_patterns = len(input)
    input = np.reshape(input, (n_patterns, seq_len, 1))
    input = input / float(vocab)
    output = np_utils.to_categorical(output)

    return input, output


def buildModel(input, vocab, learning_rate, units, drop_rate, load=None):
    model = Sequential()
    model.add(LSTM(
        units,
        input_shape=(input.shape[1], input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(drop_rate))
    model.add(LSTM(units, return_sequences=True))
    model.add(Dropout(drop_rate))
    model.add(LSTM(units))
    model.add(Dropout(drop_rate))
    model.add(Dense(vocab))
    model.add(Activation('softmax'))
    optimizer = optimizers.RMSprop(learning_rate)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer
    )
    if load:
        model.load_weights('models/{}.hdf5'.format(load))
    return model

def train(args):
    with open('notes/{}'.format(args.dataset), 'rb') as filepath:
        notes = pickle.load(filepath)

    vocab = len(set(notes))

    input, output = getXY(notes, vocab, args.seq_len)

    model = buildModel(
        input,
        vocab,
        args.learning_rate,
        args.units,
        args.drop_rate
    )  

    createDir("models")
    name = "{}-{}-{}-{}-{}".format(
        args.dataset,
        args.units,
        args.learning_rate,
        args.drop_rate,
        args.batch_size
    )
    filepath = "models/weights-"+name+"-{epoch:02d}-{loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
    )
    callbacks_list = [checkpoint]

    model.fit(
        input,
        output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks_list
    )

    return model

def getNormX(notes, pitchstrings, vocab, seq_len):
    '''
    Creates a dictionary and uses it to map notes to integers.
    Then, uses the dictionary and creates input and normalized inputs.
    '''
    note2int = dict((note, number) for number, note in enumerate(pitchstrings))

    input = []
    output = []
    for i in range(0, len(notes) - seq_len, 1):
        sequence_in = notes[i:i + seq_len]
        sequence_out = notes[i + seq_len]
        input.append([note2int[char] for char in sequence_in])
        output.append(note2int[sequence_out])

    n_patterns = len(input)

    normalized_input = np.reshape(input, (n_patterns, seq_len, 1))
    normalized_input = normalized_input / float(vocab)

    return input, normalized_input

def compose(model, input, pitchstrings, vocab, composition_len):

    start = np.random.randint(0, len(input)-1)

    int2note = dict((number, note) for number, note in enumerate(pitchstrings))

    pattern = input[start]
    output = []

    for note_index in range(composition_len):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(vocab)

        prediction = model.predict(prediction_input, verbose=0)
        selected = np.argmax(prediction)
        result = int2note[selected]
        output.append(result)

        pattern.append(selected)
        pattern = pattern[1:len(pattern)]

    return output

def createMidi(prediction_output, note_len, name):
    '''
    Generate a midi file from a melody composed by the net.
    '''
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        offset += note_len

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp='test_{}.mid'.format(name))

def generate(args):

    with open('notes/{}'.format(args.dataset), 'rb') as filepath:
        notes = pickle.load(filepath)

    pitchstrings = sorted(set(item for item in notes))
    vocab = len(set(notes))

    input, normalized_input = getNormX(notes, pitchstrings, vocab, args.seq_len)
    model = buildModel(
        normalized_input,
        vocab,
        args.learning_rate,
        args.units,
        args.drop_rate,
        args.load
    )
    prediction = compose(
        model,
        input,
        pitchstrings,
        vocab,
        args.composition_len
    )
    createMidi(prediction, args.note_len, args.dataset)

arg_parser = argparse.ArgumentParser(
    description="Model for music composition, using RNN-LSTM.")
subparsers = arg_parser.add_subparsers(title="subcommands")

encode_parser = subparsers.add_parser("encode", help="Encodes a dataset o use it for training.")
encode_parser.add_argument("--dataset", required=True,
                          help="Name of the folder inside midi that contains the dataset.")
encode_parser.set_defaults(main=encode)

train_parser = subparsers.add_parser("train", help="Trains the model with midi files.")
train_parser.add_argument("--dataset", required=True,
                          help="Name of the folder inside midi that contains the dataset.")
train_parser.add_argument("--seq-len", type=int, default=100,
                          help="Sequence length (100 by default).")
train_parser.add_argument("--units", type=int, default=512,
                          help="Units in LSTM cells (512 by default).")
train_parser.add_argument("--drop-rate", type=float, default=0.3,
                          help="Dropout rate (0.3 by default).")
train_parser.add_argument("--learning-rate", type=float, default=0.001,
                          help="Learning rate (0.001 by default).")
train_parser.add_argument("--batch-size", type=int, default=64,
                          help="Size of training batches (64 by default).")
train_parser.add_argument("--epochs", type=int, default=200,
                          help="Max number of epochs (200 by default).")
train_parser.set_defaults(main=train)



generate_parser = subparsers.add_parser("generate", help="Composes music with a trained model.")
generate_parser.add_argument("--dataset", required=True,
                          help="Name of the folder inside midi that contains the dataset.")
generate_parser.add_argument("--seq-len", type=int, default=100,
                          help="Sequence length (100 by default).")
generate_parser.add_argument("--units", type=int, default=512,
                          help="Units in LSTM cells (512 by default).")
generate_parser.add_argument("--drop-rate", type=float, default=0.3,
                          help="Dropout rate (0.3 by default).")
generate_parser.add_argument("--learning-rate", type=float, default=0.001,
                          help="Learning rate (0.001 by default).")
generate_parser.add_argument("--composition-len", type=int, default=500,
                          help="Length of the composition, in notes (500 by default).")
generate_parser.add_argument("--note-len", type=float, default=0.5,
                          help="Note length (0.5 by default).")
generate_parser.add_argument("--load", type=str, default=None,
                          help="Name of the model to load.")
generate_parser.set_defaults(main=generate)

args = arg_parser.parse_args()
args.main(args)