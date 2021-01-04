# used to create simple midis (i.e. with only five notes repeating) to test that train.py is working

import glob 
import pickle
import numpy
from music21 import converter, instrument, note, chord, stream
import tensorflow as tf
from tensorflow.keras import utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop
import random

def create_test_midi():
  offset = 0
  output_notes = []
  index = 0
  # the number of times the sequence of pitches in possible_pitches will repeat
  num_pitch_groups = 40
  # a sequence of pitches that will repeat in the file
  possible_pitches = ['G2', 'D4', 'E1', 'B3', 'C#4']
  while index < num_pitch_groups:
    for p in possible_pitches:
      new_note = note.Note(p)
      new_note.offset = offset
      new_note.storedInstrument = instrument.Piano()
      output_notes.append(new_note)

      # increase the offset to separate notes (otherwise they stack)
      offset += 0.5

    index += 1
  # converts the notes to a music21 stream object
  midi_stream = stream.Stream(output_notes)

  # writes the stream object to a file
  file_number = random.randint(0, 1000)
  PATH = './data/test' + str(file_number) + '.midi'
  midi_stream.write('midi', fp=PATH)

create_test_midi()