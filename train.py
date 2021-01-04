import glob 
import pickle
import numpy
from music21 import converter, instrument, note, chord
import tensorflow as tf
from tensorflow.keras import utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop

def get_notes():
  notes = []

  for file in glob.glob('./data/*.midi'):
    
    stream = converter.parse(file)

    print('Parsing %s' % file)

    notes_to_parse = None

    try: # if the stream has separate instruments, parse the parts separately
        s2 = instrument.partitionByInstrument(stream)
        notes_to_parse = s2.parts[0].recurse() 
    except: # if the file has notes in a flat structure, add them to note_to_parse
        notes_to_parse = stream.flat.notes

    for element in notes_to_parse:
      # if it's a note, add the note to notes
      if isinstance(element, note.Note):
        notes.append(str(element.pitch))
      # if it's a chord, add the chord in normal form
      elif isinstance(element, chord.Chord):
        notes.append('.'.join(str(n) for n in element.normalOrder))
  
  with open('./notes', 'wb') as filepath:
    pickle.dump(notes, filepath)

  print(notes)
  return notes

def prepare_sequences(notes, n_vocab):
    # the length of each sequence can be altered to improve results
    sequence_length = 35

    pitchnames = sorted(set(item for item in notes))

    # using a python dictionary to match pitchnames to integers because integers
    # work better than string-based categories 
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    # initializing the data structures holding our input and output
    # for training the NN
    network_input = []
    network_output = []

    # fills the input and output data structures
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshaping and normalizing the input
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1)) / float(n_vocab)

    # converting the integer categories to a vector of vectors containing binary values for one-hot encoding later
    network_output = utils.to_categorical(network_output)

    return (network_input, network_output)

def create_network(network_input, n_vocab):
    
    model = tf.keras.Sequential([                                 
                                  tf.keras.layers.LSTM(256, input_shape = (network_input.shape[1], network_input.shape[2]), return_sequences=True, recurrent_dropout=0.3),
                                  tf.keras.layers.LSTM(256, return_sequences=True, recurrent_dropout=0.3),
                                  tf.keras.layers.LSTM(256),
                                  tf.keras.layers.BatchNormalization(),
                                  tf.keras.layers.Dropout(0.3),
                                  tf.keras.layers.Dense(256),
                                  tf.keras.layers.ReLU(),
                                  tf.keras.layers.BatchNormalization(),
                                  tf.keras.layers.Dropout(0.3),
                                  tf.keras.layers.Dense(n_vocab, activation = 'softmax')                           
  ]
  )

    return model

def train(model, network_input, network_output):
    filepath = "./checkpoints/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=False,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=10, batch_size=128, callbacks=callbacks_list)

def train_network():
  notes = get_notes()
  n_vocab = len(set(notes))
  network_input, network_output = prepare_sequences(notes, n_vocab)

  model = create_network(network_input, n_vocab)

  model.compile(
    loss="categorical_crossentropy",
    optimizer=RMSprop(lr=0.001),
    metrics=['acc']
    )
  if not model._is_compiled:
    print('>:(')

  train(model, network_input, network_output)  

if __name__ == '__main__':
    train_network()
