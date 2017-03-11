import numpy as np

import sys

# module_path = "/home/shagun/projects/skip_thoughts"

# if module_path not in sys.path:
#     sys.path.append(module_path)

import skipthoughts

import numpy as np

def read_file(file_path):
    '''Method to read all the lines from the file into one list'''
    lines = []
    lines_set = set()
    with open(file_path) as f:
        for line in f:
            if line not in lines_set:
                lines.append(line)
    return lines

def map_lines_to_vector(lines):
    '''Method to map the list of lines into numpy array'''
    model = skipthoughts.load_model()
    vectors = skipthoughts.encode(model, lines)
    return vectors

def map_sentences_to_vectors(input_file_path="", output_file_path=""):
    '''Method to map the sentences to vectors'''
    lines = read_file(input_file_path)
    vectors = map_lines_to_vector(lines)
    with open(output_file_path + "lines.txt", "w") as f:
        for line in lines:
            f.write(line)
    np.savetxt(output_file_path + "array.npz", vectors)


if __name__ == '__main__':
    input_file_path = 'input.txt'
    map_sentences_to_vectors(input_file_path=input_file_path)