import dataclasses
from dataclasses import dataclass
import json
import os
from collections import Counter
from fractions import Fraction
from typing import Dict

import music21
from music21 import converter, note, midi, duration, meter, interval, pitch
from music21.interval import GenericInterval
from music21.meter import TimeSignature
from music21.midi import MidiFile
from music21.note import GeneralNote
from music21.stream import Measure
from music21.stream.makeNotation import consolidateCompletedTuplets

from librosa import segment

import matplotlib.pyplot as plt
import matplotlib.colors

import numpy as np
from numba.typed.dictobject import new_dict

@dataclass
class SimpleNote(object):
    pitch: int = 0
    note_duration: float = 0.0
    tie: str | None = None

    # Constructor
    # def __init__(self, pitch, note_duration, tie):
    #     self.pitch = pitch
    #     self.note_duration = note_duration
    #     self.tie = tie
    #
    # def __str__(self):
    #     return "pitch: {0}, note_duration: {1}".format(
    #         self.pitch, self.note_duration
    #     )
    
    # def __eq__(self, other):
    #     return self.pitch == other.pitch and self.note_duration == other.note_duration and self.tie == other.tie
    
@dataclass
class SimpleNotePrime(object):
    generic_interval: int = 0
    duration_ratio: float = 0.0

    # Constructor
    # def __init__(self, generic_interval, duration_ratio):
    #     self.generic_interval = generic_interval
    #     self.duration_ratio = duration_ratio
    #
    # def __str__(self):
    #     return "generic_interval: {0}, duration_ratio: {1}".format(
    #         self.generic_interval, self.duration_ratio
    #     )
    #
    def __hash__(self):
        return hash((self.generic_interval, self.duration_ratio))

    def __eq__(self, other):
        return self.generic_interval == other.generic_interval and self.duration_ratio == other.duration_ratio


# Marks a phrase's start position in terms of song number, song name, measure, and offset of measure
@dataclass
class PhrasePosition(object):
    song_index: int
    song_name: str
    track_name: str
    measure_number: int
    offset: float
    
    # def __init__(self, song_index, song_name, track_name, measure_number, offset):
    #     self.song_index = song_index
    #     self.song_name = song_name
    #     self.track_name = track_name
    #     self.measure_number = measure_number
    #     self.offset = offset


# The key that will mark a certain phrase group
@dataclass
class PhraseKey(object):
    first_position: PhrasePosition
    song_names: list[str]  # List of songs this phrase group appears in
    
    # def __init__(self, first_position, song_names):
    #     self.first_position = first_position
    #     self.song_names = song_names


# Represents a nontrivial music repetition of len(note_list)
@dataclass
class MusicString(object):
    note_list: list[SimpleNote]
    frequency: int  # Number of exact repetitions of this phrase
    positions: list[PhrasePosition]  # Position of exact repetitions
    
    # def __init__(self, note_list, frequency, positions):
    #     self.note_list = note_list
    #     self.frequency = frequency
    #     self.positions = positions
        
# A container for both the phrase_key and the music_string_list
# (When creating a dictionary, frees up the key to be a simple str or int, needed to save as JSON)
@dataclass
class PhraseGroup(object):
    phrase_key: PhraseKey
    music_string_list: list[MusicString]
    
    # def __init__(self, phrase_key, music_string_list):
    #     self.phrase_key = phrase_key
    #     self.music_string_list = music_string_list


def midi_to_measures(midi_path: str):
    # Extracts notes and rests from a MIDI file organized by track (Part).
    #
    # Parameters:
    #     midi_path (str): Path to the MIDI file.
    #
    # Returns:
    #     dict: A dictionary where each key is a track name (or index)
    #           and the value is a list of Measure objects.
    
    # Parse MIDI file
    score = converter.parse(midi_path,  quarterLengthDivisors=(2, 3, 4, 8))
    # score = score.explode()
    
    # Prepare dictionary to store information
    midi_data = {}
    
    # Time Signature is initialized by first instrument
    time_signature: TimeSignature | None = None
    
    for i, part in enumerate(score.parts):
        
        # Get track name from midi. If it doesn't have one, assign it based on the midi instrument
        track_name: str
        if part.partName and not part.partName.__contains__("Track"):
            track_name = part.partName
        else:
            track_name = part[music21.instrument.Instrument][0].__str__()
            # print(part[music21.instrument.Instrument][0])
        
        if i == 0:
            time_signature = part[music21.meter.TimeSignature][0]
        else:
            part['Measure'][0].timeSignature = time_signature
            
        # Remake measures to account for new time signature
        part = part.makeMeasures()
        
        part = part.stripTies()
        part = part.makeTies()
        
        consolidateCompletedTuplets(part, onlyIfTied=False)
        
        # Check for repetitive parts with same track_name, and rename them (for example, piano sometimes has two parts)
        same_track_number = 1
        while track_name in midi_data:
            same_track_number += 1
            track_name = track_name + " " + same_track_number.__str__()
        
        # Gets list of measures (syntax is weird because of OrderedDictionary shenanigans)
        measure_list = []
        temp = list(part.measureOffsetMap().values())
        for measures_temp in temp:
            measure = measures_temp[0]
            
            measure_list.append(measure)
        midi_data[track_name] = measure_list

    return midi_data


            
    

def measure_to_notes(measure: Measure):
    # Flatten the stream to make sure we get all notes and rests
    flat_notes = measure.flatten().notesAndRests
    return flat_notes

def measure_to_simple_note_list(measure: Measure):
    simple_note_list = []
    flat_notes = measure.flatten().notesAndRests
    
    # Within a measure, identity the melody by the highest note currently being played
    local_offset = 0
    for general_note in flat_notes:
        # Ignores all rests
        # To-fix: figure out best way to deal with note intervals at the end of phrases / before rests.
        if not general_note.isRest and general_note.offset >= local_offset:
            # Get the highest pitch of chord or note
            pitch_list = [p.midi for p in general_note.pitches]
            highest_pitch = 0
            for pitch in pitch_list:
                if pitch > highest_pitch:
                    highest_pitch = pitch
                    
            # Add new simple note to list
            if general_note.tie is None:
                tie_type = None
            else:
                tie_type = general_note.tie.type
            simple_note = SimpleNote(highest_pitch, general_note.quarterLength, tie_type)
            simple_note_list.append(simple_note)
            
            # Increment local_offset to prevent note overlap
            local_offset += general_note.quarterLength
            
    return simple_note_list

def simple_notes_to_simple_primes(simple_measures: list[list[SimpleNote]]):
    measures_prime = []
    
    previous_note: SimpleNote
    previous_note_measure: int
    current_note: SimpleNote = SimpleNote(pitch=0, note_duration=0.0, tie=None)
    current_note_measure: int = 0
    for measure_number, simple_measure in enumerate(simple_measures):
        # Creates new list for current measure
        measures_prime.append([])
        for note_index, simple_note in enumerate(simple_measure):
            # Skips rest of code for the first note so that we have all the values we need
            if current_note.pitch == 0:
                current_note = simple_note
                current_note_measure = measure_number
                continue
            
            # Creates a snp for the previous_note by keeping track of that note's position in the list
            previous_note = current_note
            previous_note_measure = current_note_measure
            
            current_note = simple_note
            current_note_measure = measure_number
            
            p1 = pitch.Pitch(previous_note.pitch)
            p2 = pitch.Pitch(current_note.pitch)
            a_interval = interval.Interval(pitchStart=p1, pitchEnd=p2)
            generic_interval = a_interval.generic.directed
            
            duration_ratio = current_note.note_duration / previous_note.note_duration
            
            simple_note_prime = SimpleNotePrime(generic_interval, duration_ratio)
            measures_prime[previous_note_measure].append(simple_note_prime)
        
    return measures_prime

def calculate_dice_coefficient(measure_prime_1: list[SimpleNotePrime], measure_prime_2: list[SimpleNotePrime]):
    intersection_set = list((Counter(measure_prime_1) & Counter(measure_prime_2)).elements())
    
    if len(measure_prime_1) == 0 or len(measure_prime_2) == 0:
        return 0.0
    else:
        return 2 * len(intersection_set) / (len(measure_prime_1) + len(measure_prime_2))

def create_self_similarity_matrix(measures_prime: list[list[SimpleNotePrime]]) -> list[list[float]]:
    length = len(measures_prime)
    self_similarity_matrix = [[0.0 for i in range(length)] for j in range(length)]
    for index1, measure1 in enumerate(measures_prime):
        for index2, measure2 in enumerate(measures_prime):
            self_similarity_matrix[index1][index2] = calculate_dice_coefficient(measure1, measure2)
    return self_similarity_matrix


def create_boolean_ssm(self_similarity_matrix: list[list[float]], threshold: float) -> list[list[bool]]:
    boolean_ssm: list[list[bool]] = []
    for row in self_similarity_matrix:
        boolean_ssm.append([bool(x > threshold) for x in row])
    return boolean_ssm

# Create a lag matrix, which is a representation of a ssm where the diagonals are turned into rows
# Only uses the bottom-left half of the ssm, since the other half is repeat information
def create_lag_matrix(boolean_ssm: list[list[bool]]):
    size = len(boolean_ssm)
    lag_matrix = [[False for i in range(size)] for j in range(size)]
    for i in range(size):
        for j in range(size):
            if i+j < size:
                lag_matrix[i][j] = boolean_ssm[i+j][j]
            else:
                break
    return lag_matrix

def find_note_list_by_measure_range(simple_note_lists: list[list[SimpleNote]], measure_range: range):
    note_list: list[SimpleNote] = []
    for measure_number in measure_range:
        #TO-FIX: Reduce range so that it can't (somehow) be outside of range
        if measure_number >= len(simple_note_lists):
            continue
        
        note_list.extend(simple_note_lists[measure_number])
    return note_list
        

# The phrase dictionary stores all the repeated motifs as MusicString objects
# - The key is a PhraseKey object, which stores prominent information about a motif group
# - The value is a list of all motifs that are similar to each other (but not exactly the same)
# The motifs are found from the lag_matrix, and the notes are identified from the simple_midi_data
def create_phrase_dictionary():
    phrase_dictionary: Dict[int, PhraseGroup] = {}
    return phrase_dictionary


# Adds an entire song (including its tracks) to the phrase dictionary
def add_to_phrase_dictionary(phrase_dictionary: Dict[int, PhraseGroup], song_index: int,
                             song_name: str, lag_matrix_list: list[list[list[bool]]],
                             simple_midi_data: Dict[str, list[list[SimpleNote]]]):
    
    index = 0
    for track, simple_note_lists in simple_midi_data.items():
        
        lag_matrix = lag_matrix_list[index]
        note_list_length = len(simple_note_lists)
        print(song_name + " / " + track)
        
        #print("SNL: " + len(simple_note_lists).__str__())

        for row_index, lag_row in enumerate(lag_matrix):
            print("Row: " + row_index.__str__())
            # Skip first row
            if row_index == 0:
                continue
                
            # Skip rows after list length
            elif row_index >= note_list_length:
                continue
                
            # Variables to find repeated phrase
            found_phrase: bool = False  # A new phrase is found and being measured between cells
            starting_measure_1: int = 0  # First starting measure of the phrase
            starting_measure_2: int = 0  # Second starting measure of the phrase
            phrase_length: int = 0  # Length in measures
            for column_index, cell in enumerate(lag_row):
                # print("Column: " + column_index.__str__())
                
                # Skip columns after list length
                if column_index >= note_list_length:
                    continue
                
                if cell:
                    if not found_phrase:
                        found_phrase = True
                        starting_measure_1 = column_index
                        starting_measure_2 = column_index + row_index
                        phrase_length = 0
                    phrase_length += 1
                    
                # End current phrase and create MusicString for it.
                # - If the MusicString is new, then add a new entry to the dictionary
                # - Else, update existing motif group accordingly
                elif found_phrase:
                    found_phrase = False
                    
                    # PhrasePosition objects
                    first_position = PhrasePosition(song_index, song_name, track, starting_measure_1, 0)
                    second_position = PhrasePosition(song_index, song_name, track, starting_measure_2, 0)
                    
                    # Search for pre-existing match in phrase dictionary
                    phrase_match_found = False
                    
                    for int_key, phrase_group in phrase_dictionary.items():
                        phrase_key = phrase_group.phrase_key
                        music_string_list = phrase_group.music_string_list
                        
                        # Look for matches within the same song
                        if phrase_key.song_names.__contains__(song_name):
                            
                            # Look for matches within the same track
                            if phrase_key.first_position.track_name == track:
                                
                                exact_match = False
                                similar_match = False
                                chosen_music_string: MusicString | None = None
                                note_list: list[SimpleNote] = []
                                chosen_position: PhrasePosition | None = None
                                
                                
                                for music_string in music_string_list:
                                    
                                    # Find a match (either exact or similar)
                                    match_first_position = music_string.positions.__contains__(first_position)
                                    match_second_position = music_string.positions.__contains__(second_position)
                                    
                                    # Ignore if both positions are already in music_string
                                    if match_first_position and match_second_position:
                                        continue
                                    
                                    # If only one position is accounted for, then process the other
                                    # (Steps change based on whether the match is exact or similar)
                                    elif match_first_position or match_second_position:
                                        
                                        # Recreate MusicString for each position to determine exact or similar
                                        if match_first_position:
                                            range_2 = range(starting_measure_2, starting_measure_2 + phrase_length)
                                            note_list = find_note_list_by_measure_range(simple_note_lists, range_2)
                                            chosen_position = second_position
                                        else:
                                            range_1 = range(starting_measure_1, starting_measure_1 + phrase_length)
                                            note_list = find_note_list_by_measure_range(simple_note_lists, range_1)
                                            chosen_position = first_position
                                        
                                        # print(note_list)
                                        # print("")
                                        if music_string.note_list.__eq__(note_list):
                                            chosen_music_string = music_string
                                            exact_match = True
                                            
                                            # If an exact match is found, then no need to search further
                                            break
                                        else:
                                            similar_match = True
                                
                                    
                                # If there is an exact match, update positions list and frequency
                                if exact_match:
                                    # Update chosen position
                                    chosen_music_string.positions.append(chosen_position)
                                    
                                    # Update frequency
                                    chosen_music_string.frequency += 1
                                
                                # If there is a similar match, create new MusicString in phrase group
                                elif similar_match:
                                    new_music_string = MusicString(note_list, 1, [chosen_position])
                                    music_string_list.append(new_music_string)
                            
                            # Currently, ignore potential matches within other tracks
                            else:
                                continue
                        
                    # If no matches are found, then create new phrase group
                    if not phrase_match_found:
                        
                        # Create phrase_key
                        phrase_key = PhraseKey(first_position, [song_name])
                        
                        # Create MusicString objects
                        # - One if the phrases are exact, and two if they are similar
                        range_1 = range(starting_measure_1, starting_measure_1 + phrase_length)
                        first_note_list = find_note_list_by_measure_range(simple_note_lists, range_1)
                        
                        range_2 = range(starting_measure_2, starting_measure_2 + phrase_length)
                        second_note_list = find_note_list_by_measure_range(simple_note_lists, range_2)
                        
                        new_music_string_list: list[MusicString] = []
                        
                        if first_note_list.__eq__(second_note_list):
                            new_music_string = MusicString(first_note_list, 2, [first_position, second_position])
                            new_music_string_list.append(new_music_string)
                        else:
                            first_music_string = MusicString(first_note_list, 1, [first_position])
                            second_music_string = MusicString(second_note_list, 1, [second_position])
                            new_music_string_list.append(first_music_string)
                            new_music_string_list.append(second_music_string)
                            
                        # Add new entry to dictionary
                        new_phrase_group = PhraseGroup(phrase_key, new_music_string_list)
                        new_int_key = len(list(phrase_dictionary.keys()))
                        
                        phrase_dictionary[new_int_key] = new_phrase_group
                        

        

def note_to_string(general_note: GeneralNote):
    txt = ""
    if general_note.isNote:
        txt = "type: {0}, pitch: {1}, midi_pitch: {2}, duration_quarter: {3}, offset: {4}, velocity: {5}".format(
            "note", general_note.pitches[0].nameWithOctave, general_note.pitches[0].midi,
            general_note.quarterLength, general_note.offset, general_note.volume.velocity
        )
    elif general_note.isChord:
        txt = "type: {0}, pitches: {1}, midi_pitch: {2}, duration_quarter: {3}, offset: {4}, velocity: {5}".format(
            "chord", [p.nameWithOctave for p in general_note.pitches], [p.midi for p in general_note.pitches],
            general_note.quarterLength, general_note.offset, general_note.volume.velocity
        )
    elif general_note.isRest:
        txt = "type: {0}, duration_quarter: {1}, offset: {2}".format(
            "rest", general_note.quarterLength, general_note.offset
        )
        
    return txt
    
def print_notes(midi_data):
    for track, measures in midi_data.items():
        print(track)
        for measure_number, measure in enumerate(measures):
            print("Measure: " + measure_number.__str__())
            for index, note in enumerate(measure.notesAndRests):
                print(index.__str__() + " " + note_to_string(note))
                
def print_prime_notes(simple_midi_prime_data: Dict[str, list[list[SimpleNotePrime]]]):
    for track, note_prime_lists in simple_midi_prime_data.items():
        print(track)
        for measure_number, note_prime_list in enumerate(note_prime_lists):
            print("Measure: " + measure_number.__str__())
            for index, note_prime in enumerate(note_prime_list):
                print(index.__str__() + " " + note_prime.__str__())
                
                
def plot_colored_grid(data, song_name, track_name):
    # Color for False and True
    cmap = matplotlib.colors.ListedColormap(['black', 'white'])
    
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['savefig.dpi'] = 200
    plt.imshow(data, cmap=cmap)
    plt.title("{0}: {1}".format(song_name, track_name))
    plt.show()
    
    
    
def process_midi_file(midi_filepath: str):
    midi_data = midi_to_measures(midi_filepath)
    
    simple_midi_data: Dict[str, list[list[SimpleNote]]] = {}
    for track, measures in midi_data.items():
        simple_midi_data[track] = []
        for index, measure in enumerate(measures):
            data = measure_to_simple_note_list(measure)
            simple_midi_data[track].append(data)

    simple_midi_prime_data: Dict[str, list[list[SimpleNotePrime]]] = {}
    for track, simple_note_lists in simple_midi_data.items():
        simple_midi_prime_data[track] = simple_notes_to_simple_primes(simple_note_lists)
    
    # Print self-similarity matrix for each track
    track_ssm_list: list[list[list[float]]] = []
    for track in list(simple_midi_prime_data.keys()):
        self_similarity_matrix = create_self_similarity_matrix(simple_midi_prime_data[track])
        track_ssm_list.append(self_similarity_matrix)

    threshold: float = 0.7
    boolean_ssm_list: list[list[list[bool]]] = []
    for ssm in track_ssm_list:
        boolean_ssm = create_boolean_ssm(ssm, threshold)
        boolean_ssm_list.append(boolean_ssm)

    lag_matrix_list: list[list[list[bool]]] = []
    for boolean_ssm in boolean_ssm_list:
        lag_matrix = create_lag_matrix(boolean_ssm)
        lag_matrix_list.append(lag_matrix)
    
    song_name = os.path.basename(midi_filepath)
    # - Uncomment to see SSM and Lag Matrix graphs
    song_name = "Space Junk Road"
    # Draw First Graph
    first_track = list(simple_midi_prime_data.keys())[0]
    plot_colored_grid(boolean_ssm_list[0], song_name, first_track)
    plot_colored_grid(lag_matrix_list[0], song_name, first_track)
    
    # Draw All Graphs
    # for index, track in enumerate(list(simple_midi_prime_data.keys())):
    #     plot_colored_grid(boolean_ssm_list[index], song_name, track)
    #     plot_colored_grid(lag_matrix_list[index], song_name, track)
    
    return simple_midi_data, lag_matrix_list

def test_single_file():
    midi_file = "MidiFiles/superMarioGalaxy/spacejunk.mid"
    process_midi_file(midi_file)
    
    
def test_multiple_files():
    folder_name = "MidiFiles/superMarioGalaxy"
    midi_files = []
    for (dirpath, dirnames, filenames) in os.walk(folder_name):
        midi_files.extend(filenames)
    
    for midi_file in midi_files:
        process_midi_file(folder_name + "/" + midi_file)
     
     
class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)
        
def process_phrase_dictionary():
    # Create phrase Dictionary
    phrase_dictionary = create_phrase_dictionary()

    # Setup up code to read each midi file for Super Mario Galaxy
    folder_name = "MidiFiles/superMarioGalaxy"
    game_name = "Super Mario Galaxy"

    midi_files = []
    for (dirpath, dirnames, filenames) in os.walk(folder_name):
        midi_files.extend(filenames)

    for temp_song_index, midi_file in enumerate(midi_files):

        song_name = midi_file
        simple_midi_data, lag_matrix_list = process_midi_file(folder_name + "/" + midi_file)

        # values = list(simple_midi_data.values())
        # for value in values:
        #     print(len(value))

        add_to_phrase_dictionary(phrase_dictionary, temp_song_index, song_name, lag_matrix_list, simple_midi_data)

    json_str = json.dumps(phrase_dictionary, indent=4, cls=EnhancedJSONEncoder)
    with open("SuperMarioGalaxy.json", "w") as f:
        f.write(json_str)
    
    

if __name__ == "__main__":
    # Separate Tests
    test_single_file()
    # test_multiple_files()
    
    # process_phrase_dictionary()

    # Print structured summary
    # print_prime_notes(simple_midi_prime_data)
