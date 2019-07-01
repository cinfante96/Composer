import glob
import pickle
import argparse
import os
import json
import numpy as np
from clr_callback import CyclicLR
from random import shuffle, randint, choice
from collections import OrderedDict
from music21 import converter, instrument, note, chord, stream
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import Activation
from keras.utils import np_utils
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras import optimizers
import matplotlib.pyplot as plt

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

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
                notes.append(
                    '.'.join(str(n) for n in element.normalOrder)
                )
    createDir("notes")
    fname = 'notes/{}'.format(dataset)
    if os.path.exists(fname):
        print("Encoded dataset already exists.")
    else:
        with open(fname, 'wb') as filepath:
            pickle.dump(notes, filepath)
            print("Encoded dataset saved in ",fname)
    createDir("dicts")
    fname = 'dicts/{}-note2int'.format(dataset)
    pitchstrings = sorted(set(item for item in notes))
    shuffle(pitchstrings)
    if os.path.exists(fname):
        print("Note2Int dict already exists.")
    else:
        note2int = dict(
            (note, number) for number, note in enumerate(pitchstrings)
        )
        with open(fname, 'wb') as filepath:
            pickle.dump(note2int, filepath)
            print("Int2Note dict saved in ",fname)
    fname = 'dicts/{}-int2note'.format(dataset)
    if os.path.exists(fname):
        print("Int2Note dict already exists.")
    else:
        int2note = dict(
            (number, note) for number, note in enumerate(pitchstrings)
        )
        with open(fname, 'wb') as filepath:
            pickle.dump(int2note, filepath)
            print("Int2Note dict saved in ",fname)

    return notes


def getXY(notes, note2int, vocab, seq_len):
    '''
    Creates a dictionary and uses it to map integers to notes.
    Then, uses the dictionary and creates normalized input and output for the network.
    '''
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

def buildModelLSTM(input, vocab, learning_rate, units, drop_rate, num_layers, load=None):
    model = Sequential()

    if num_layers == 1:
        model.add(LSTM(
        units,
        input_shape=(input.shape[1], input.shape[2]),
        return_sequences=False
        ))
        model.add(Dropout(drop_rate))
        model.add(Dense(vocab))
        model.add(Activation('softmax'))
    elif num_layers == 2:
        model.add(LSTM(
        units,
        input_shape=(input.shape[1], input.shape[2]),
        return_sequences=True
        ))
        model.add(Dropout(drop_rate))
        model.add(LSTM(units))
        model.add(Dropout(drop_rate))
        model.add(Dense(vocab))
        model.add(Activation('softmax'))
    else:
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
        optimizer=optimizer,
        metrics=['acc']
    )
    if load:
        model.load_weights('models/{}.hdf5'.format(load))
    return model

def buildModelRNN(input, vocab, learning_rate, units, drop_rate, num_layers, load=None):
    model = Sequential()

    if num_layers == 1:
        model.add(SimpleRNN(
        units,
        input_shape=(input.shape[1], input.shape[2]),
        return_sequences=False
        ))
        model.add(Dropout(drop_rate))
        model.add(Dense(vocab))
        model.add(Activation('softmax'))
    elif num_layers == 2:
        model.add(SimpleRNN(
        units,
        input_shape=(input.shape[1], input.shape[2]),
        return_sequences=True
        ))
        model.add(Dropout(drop_rate))
        model.add(SimpleRNN(units))
        model.add(Dropout(drop_rate))
        model.add(Dense(vocab))
        model.add(Activation('softmax'))
    else:
        model.add(SimpleRNN(
        units,
        input_shape=(input.shape[1], input.shape[2]),
        return_sequences=True
        ))
        model.add(Dropout(drop_rate))
        model.add(SimpleRNN(units, return_sequences=True))
        model.add(Dropout(drop_rate))
        model.add(SimpleRNN(units))
        model.add(Dropout(drop_rate))
        model.add(Dense(vocab))
        model.add(Activation('softmax'))

    optimizer = optimizers.RMSprop(learning_rate)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['acc']
    )
    if load:
        model.load_weights('models/{}.hdf5'.format(load))
    return model

def train(args):
    notes = readMids(args.dataset)

    with open('dicts/{}-note2int'.format(args.dataset),'rb') as filepath:
        note2int = pickle.load(filepath)

    vocab = len(set(notes))

    input, output = getXY(notes, note2int, vocab, args.seq_len)

    if args.rnn:
        model = buildModelRNN(
            input,
            vocab,
            args.learning_rate,
            args.units,
            args.drop_rate,
            args.num_layers
        )  
    else:
        model = buildModelLSTM(
            input,
            vocab,
            args.learning_rate,
            args.units,
            args.drop_rate,
            args.num_layers
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
    #if args.plot_model_graph:
        #plot_model(model, to_file="models/weights-"+name+"-{epoch:02d}-{loss:.2f}.png")
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
    )
    if args.cyclic_learning_rate:

        clr = CyclicLR(base_lr=0.001, max_lr=0.03, step_size=11100, mode='triangular')
        callbacks_list = [checkpoint,clr]

        history = model.fit(
            input,
            output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks_list
          )

        h = clr.history
        lr = h['lr']
        #acc = h['acc']
        lss = h['loss']

        moving_average_lr = running_mean(np.array(lr),400)
        #moving_average_acc = running_mean(np.array(acc),400)
        moving_average_lss = running_mean(np.array(lss),400)

        plt.plot(lr,lss)
        plt.ylabel('Error')
        plt.xlabel('Tasa de Aprendizaje')
        plt.show()

        plt.plot(moving_average_lr.tolist(),moving_average_lss.tolist())
        plt.ylabel('Error')
        plt.xlabel('Tasa de Aprendizaje')
        plt.show()

    else:
        callbacks_list = [checkpoint]

        history = model.fit(
                    input,
                    output,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    callbacks=callbacks_list
                  )
    
    if args.save_history:
        with open("history/"+name,'a') as f:
            json.dump(history.history['loss'],f)

    if args.plot_history:
        plt.plot(history.history['loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train'], loc='upper left')
        plt.show()

    return model

def getNormX(notes, note2int, vocab, seq_len):
    '''
    Creates a dictionary and uses it to map notes to integers.
    Then, uses the dictionary and creates input and normalized inputs.
    '''

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

def compose(model, input, vocab, composition_len):

    start = np.random.randint(0, len(input)-1)

    with open('dicts/{}-int2note'.format(args.dataset),'rb') as filepath:
        int2note = pickle.load(filepath)

    pattern = input[start]
    output = []

    for _ in range(composition_len):
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

    if args.display_sheet:
        midi_stream.show('lily')

def generate(args):

    with open('notes/{}'.format(args.dataset), 'rb') as filepath:
        notes = pickle.load(filepath)

    with open('dicts/{}-note2int'.format(args.dataset),'rb') as filepath:
        note2int = pickle.load(filepath)

    vocab = len(set(notes))

    input, normalized_input = getNormX(notes, note2int, vocab, args.seq_len)
    if args.rnn:
        model = buildModelRNN(
            normalized_input,
            vocab,
            args.learning_rate,
            args.units,
            args.drop_rate,
            args.num_layers,
            args.load
        )
    else:
        model = buildModelLSTM(
            normalized_input,
            vocab,
            args.learning_rate,
            args.units,
            args.drop_rate,
            args.num_layers,
            args.load
        )
    with open('notes/{}'.format(args.seed), 'rb') as filepath:
        seed_notes = pickle.load(filepath)
    seed, _ = getNormX(seed_notes, note2int, vocab, args.seq_len)
    prediction = compose(
        model,
        seed,
        vocab,
        args.composition_len
    )
    createMidi(prediction, args.note_len, args.dataset)

def randomize(args):

    with open('dicts/{}-int2note'.format(args.dataset),'rb') as filepath:
        int2note = pickle.load(filepath)
    numbers = [randint(0,len(int2note)-1) for _ in range(args.composition_len)]
    notes = list(map(lambda n: int2note[n],numbers))
    createMidi(notes, args.note_len, args.dataset)

def markov(args):

    notes = readMids(args.dataset)

    with open('dicts/{}-note2int'.format(args.dataset),'rb') as filepath:
        note2int = pickle.load(filepath)

    vocab = len(set(notes))
    input = []
    output = []
    seq_len = args.seq_len
    for i in range(0, len(notes) - seq_len, 1):
        sequence_in = notes[i:i + seq_len]
        sequence_out = notes[i + seq_len]
        input.append([note2int[char] for char in sequence_in])
        output.append(note2int[sequence_out])

    classes = set(map(repr,input))
    preds = [[] for _ in range(len(classes))]
    ngrams = dict(zip(classes,preds))
    for i,o in zip(input,output):
        ngrams[repr(i)].append(o)
    
    with open('dicts/{}-int2note'.format(args.dataset),'rb') as filepath:
        int2note = pickle.load(filepath)    
    
    composition = []
    state = choice(input) # select random first state
    for _ in range(args.composition_len):
        pred = ngrams.get(repr(state))
        if pred:
            next = choice(pred)
            composition.append(int2note[next])
            state = state[1:] + [next]
        else:
            print("Unknown sequence.")
            break
    createMidi(composition, args.note_len, args.dataset)


def display(args):

    path = os.path.abspath('./{}'.format(args.midi_path))
    midi_stream = converter.parseFile(path)
    notes_to_parse = None

    try:
        score = instrument.partitionByInstrument(midi_stream)
        notes_to_parse = score.parts[0].recurse() 
    except:
        notes_to_parse = midi_stream.flat.notes

    notes_to_parse.show('lily')

arg_parser = argparse.ArgumentParser(
    description="Model for music composition, using RNN-LSTM.")
subparsers = arg_parser.add_subparsers(title="subcommands")

# Subparser for training.
train_parser = subparsers.add_parser("train", help="Trains the model with midi files.")
train_parser.add_argument("--dataset", required=True,
                          help="Name of the folder inside midi that contains the dataset.")
train_parser.add_argument("--seq_len", type=int, default=100,
                          help="Sequence length (100 by default).")
train_parser.add_argument("--units", type=int, default=512,
                          help="Units in LSTM cells (512 by default).")
train_parser.add_argument("--drop_rate", type=float, default=0.3,
                          help="Dropout rate (0.3 by default).")
train_parser.add_argument("--learning_rate", type=float, default=0.001,
                          help="Learning rate (0.001 by default).")
train_parser.add_argument("--batch_size", type=int, default=64,
                          help="Size of training batches (64 by default).")
train_parser.add_argument("--epochs", type=int, default=200,
                          help="Max number of epochs (200 by default).")
train_parser.add_argument("--num_layers", type=int, default=1, help="Number of LSTM layers in the model (1 by default).")
train_parser.add_argument("-r","--rnn", action="store_true", help="Trains with a simple RNN, instead of a LSTM network.")
train_parser.add_argument("-g","--plot_model_graph", action="store_true", help="This will plot a graph of the model and save it to a file.")
train_parser.add_argument("-p","--plot_history", action="store_true", help="This will create a plot for the training history of the model.")
train_parser.add_argument("-c","--cyclic_learning_rate", action="store_true", help="This will use Cyclic Learning Rate to find an optimal learning rate for the model.")
train_parser.add_argument("-s","--save_history", action="store_true", help="Saves the training history in .json format.")
train_parser.set_defaults(main=train)

# Subparser for generating music.
generate_parser = subparsers.add_parser("generate", help="Composes music with a trained model.")
generate_parser.add_argument("--dataset", required=True,
                          help="Name of the folder inside midi that contains the dataset.")
generate_parser.add_argument("--seed", required=True,
                          help="Name of the folder inside midi that contains the seed dataset.")
generate_parser.add_argument("--seq_len", type=int, default=100,
                          help="Sequence length (100 by default).")
generate_parser.add_argument("--units", type=int, default=512,
                          help="Units in LSTM cells (512 by default).")
generate_parser.add_argument("--drop_rate", type=float, default=0.3,
                          help="Dropout rate (0.3 by default).")
generate_parser.add_argument("--learning_rate", type=float, default=0.001,
                          help="Learning rate (0.001 by default).")
generate_parser.add_argument("--composition_len", type=int, default=500,
                          help="Length of the composition, in notes (500 by default).")
generate_parser.add_argument("--note_len", type=float, default=0.5,
                          help="Note length (0.5 by default).")
generate_parser.add_argument("--load", type=str, default=None,
                          help="Name of the model to load.")
generate_parser.add_argument("--num_layers", type=int, default=1, help="Number of LSTM layers in the model (1 by default).")
generate_parser.add_argument("-d","--display_sheet", action="store_true", help="Displays a music sheet image of the generated melody.")
generate_parser.add_argument("-r","--rnn", action="store_true", help="Generate music with a simple RNN, instead of a LSTM network.")
generate_parser.set_defaults(main=generate)

# Subparser for generating random music.
generate_parser = subparsers.add_parser("randomize", help="Composes music randomly.")
generate_parser.add_argument("--dataset", required=True,
                          help="Name of the folder inside midi that contains the dataset.")
generate_parser.add_argument("--composition_len", type=int, default=500,
                          help="Length of the composition, in notes (500 by default).")
generate_parser.add_argument("--note_len", type=float, default=0.5,
                          help="Note length (0.5 by default).")
generate_parser.add_argument("-d","--display_sheet", action="store_true", help="Displays a music sheet image of the generated melody.")
generate_parser.set_defaults(main=randomize)

# Subparser for composing with markov chains and n-grams.
generate_parser = subparsers.add_parser("markov", help="Composes music using markov chains and n-grams.")
generate_parser.add_argument("--dataset", required=True,
                          help="Name of the folder inside midi that contains the dataset.")
generate_parser.add_argument("--composition_len", type=int, default=500,
                          help="Length of the composition, in notes (500 by default).")
generate_parser.add_argument("--note_len", type=float, default=0.5,
                          help="Note length (0.5 by default).")
generate_parser.add_argument("--seq_len", type=int, default=8,
                          help="Sequence length (8 by default).")
generate_parser.add_argument("-d","--display_sheet", action="store_true", help="Displays a music sheet image of the generated melody.")
generate_parser.set_defaults(main=markov)


# Subparser for displaying music sheets.
generate_parser = subparsers.add_parser("display", help="Display a music sheet image for a midi.")
generate_parser.add_argument("--midi_path",required=True,help="Path to the midi.")
generate_parser.set_defaults(main=display)

args = arg_parser.parse_args()
args.main(args)