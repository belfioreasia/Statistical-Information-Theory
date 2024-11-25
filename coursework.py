# Statistical Information Theory Coursework
# Belfiore Asia, cid: 02129867
# November 24th, 2024, Imperial Collenge London
"""
This file is part of Lab 4 (Hamming Codes), assessed coursework for the module
COMP70103 Statistical Information Theory. 

You should submit an updated version of this file, replacing the
NotImplementedError's with the correct implementation of each function. Do not
edit any other functions.

Follow further instructions in the attached .pdf and .ipynb files, available
through Scientia.
"""
from typing import Callable, Dict, List, Tuple, Union
import numpy as np
import numpy.random as rn
from itertools import product
from pandas.core.common import flatten

alphabet = "abcdefghijklmnopqrstuvwxyz01234567890 .,\n"
digits = "0123456789"

def char2bits(char: chr) -> np.array:
    '''
    Given a character in the alphabet, returns a 8-bit numpy array of 0,1 which represents it
    '''
    num   = ord(char)
    if num >= 256:
        raise ValueError("Character not recognised.")
    bits = format(num, '#010b')
    bits  = [ int(b) for b in bits[2:] ]
    return np.array(bits)


def bits2char(bits) -> chr:
    '''
    Given a 7-bit numpy array of 0,1 or bool, returns the character it encodes
    '''
    bits  = ''.join(bits.astype(int).astype(str))
    num   =  int(bits, base = 2)
    return chr(num)


def text2bits(text: str) -> np.ndarray:
    '''
    Given a string, returns a numpy array of bool with the binary encoding of the text
    '''
    text = text.lower()
    text = [ t for t in text if t in alphabet ]
    bits = [ char2bits(c) for c in text ]
    return np.array(bits, dtype = bool).flatten()


def bits2text(bits: np.ndarray) -> str:
    '''
    Given a numpy array of bool or 0 and 1 (as int) which represents the
    binary encoding a text, return the text as a string
    '''
    if np.mod(len(bits), 8) != 0:
        raise ValueError("The length of the bit string must be a multiple of 8.")
    bits = bits.reshape(int(len(bits)/8), 8)
    chrs = [ bits2char(b) for b in bits ]
    return ''.join(chrs)


def parity_matrix(m : int) -> np.ndarray:
    """
    Given a number of parity bits, returns the corresponding parity check matrix
    
    m : int
      The number of parity bits to use

    return : np.ndarray
      m-by-n parity check matrix
    """
    matrix_rows = []
    n = 2**m - 1

    # generate binary representation of each number <= n (with n=number of matrix columns)
    n_binary_encodings = [np.binary_repr(i+1, width=m) for i in range(n)]
    n_binary_encodings = [[int(c[i]) for i in range(m)] for c in n_binary_encodings]

    for i in range(m-1, -1, -1):
      matrix_rows.append([n_binary_encodings[j][i] for j in range(2**m-1)])

    H = np.array(matrix_rows)

    return H


def hamming_generator(m : int) -> np.ndarray:
    """
    Given a number of parity bits, returns the corresponding generator matrix
    
    m : int
      The number of parity bits to use

    return : np.ndarray
      k-by-n generator matrix
    """
    H = parity_matrix(m)
    k = (2**m) - m - 1

    transpose_rows = []

    parity_bit_positions = [((2**i)-1) for i in range(m)]
    identity_k = np.identity(k) # for columns in position of parity bits
    identity_idx = parity_idx = 0 

    for row_idx in range(k+m): 
      row = []
      if row_idx in parity_bit_positions:
        parity_bits_from_H = H[:][parity_idx]
        for bit in range(H.shape[1]):
          if bit not in parity_bit_positions:
            row.append(parity_bits_from_H[bit])
        parity_idx += 1
      else:
        row = [int(identity_row) for identity_row in identity_k[identity_idx].tolist()]
        identity_idx += 1

      transpose_rows.append(row)

    G = np.array(transpose_rows).T

    return G


def hamming_encode(data : np.ndarray, m : int) -> np.ndarray:
    """
    Given a binary data block and a suitable number of parity bits, encode the 
    data block using Hamming code from the given number of parity bits
    NB: The encoding will only work if the number of parity bits is correct for the data block
        i.e. if the size of the data block is 2^m - m - 1. Otherwise an error will be raised.

    data : np.ndarray
      array of shape (k,) with the block of bits to encode

    m : int
      The number of parity bits to use

    return : np.ndarray
      array of shape (n,) with the corresponding Hamming codeword
    # """
    # NB: need to ensure that the number of parity bits is correct for the data block
    assert (data.shape[0] == 2**m - m - 1), f"Inputs {data.shape[0], m} don't match the required sizes"

    G = hamming_generator(m)
    t = (G.T @ data) % 2 # transmitted string

    return t


def hamming_decode(code : np.ndarray, m : int) -> np.ndarray:
    """
    Given a binary codeword and a suitable number of parity bits, decode the 
    codeword using Hamming code from the given number of parity bits

    code : np.ndarray
      Array of shape (n,) containing a Hamming codeword computed with m parity bits
    m : int
      Number of parity bits used when encoding

    return : np.ndarray
      Array of shape (k,) with the decoded and corrected data
    """
    assert (np.log2(len(code) + 1) == int(np.log2(len(code) + 1)) == m), f"Inputs {len(code), m} don't match the required sizes"

    H = parity_matrix(m)
    z = (H @ code) % 2 # syndrome

    if (z>0).any(): # if there is an error
      z_binary = [int(i) for i in z]
      error_pos = -1
      for i in range(len(z_binary)):
        error_pos += (z_binary[i] * (2**i))
      # print(f"Error detected in position {error_pos+1}")
      # print(f"Received message: {code}")
      code = flip_bit(code, error_pos)
      # print(f"Corrected message: {code}")   

    R = np.copy(hamming_generator(m))
    parity_bits = [((2**i)-1) for i in range(m)]
    for i in parity_bits:
      R[:,i] = np.zeros(R.shape[0]).T
    decoded_string = (R@code) % 2 

    return decoded_string


def decode_secret(msg : np.ndarray) -> str:
    """
    Given a binary secret message, find a suitable number of parity bits 
    and decode the message using the resulting Hamming code.
    Furthermore, check if the decoded message is a valid English text.

    msg : np.ndarray
      One-dimensional array of binary integers

    return : str
      String with decoded text
    """    
    # Since I assume every character of the secret message has been
    # encoded with 8 bits (followning the text2bit function above)
    # the number of parity bits needed for each character is 4
    # following the formula 2^m ≥ k + m + 1
    # m = 4 <-- Your guess goes here

    decoded_msg = {}
    for m in range(1,11): # try possible m values up to 10
      n = 2**m - 1
      if len(msg) % n == 0: # to create chunks of n-sized codewords
        # print(f"m={m}, n={n}")
        decoded_string = []
        i = 0 # number of n-shaped blocks
        done = False
        while not done:
          # divide message into n-sized chunks
          start = i*n
          end = start + n
          codeword = msg[start:end].copy()
          source_string = hamming_decode(codeword, m).tolist()
          decoded_string.append(source_string)
          i += 1
          if end == len(msg): 
            done = True
        decoded_msg[m] = decoded_string
      else: 
        # print(f"m={m} not valid")
        continue
        
    # assuming the secret message is in english, check
    # if the decoded message is valid english text
    for m in decoded_msg.keys():
      try:
        string_code = np.array(decoded_msg[m]).flatten()
        decoded_secret = bits2text(string_code)
        if is_valid_text(decoded_secret):
          # print(f"Decoded with m={m}:\n")
          return decoded_secret
      except:
        continue

    return "No valid decoding found."


def binary_symmetric_channel(data : np.ndarray, p : float) -> np.ndarray:
    """
    Given a binary codeword and a noise level p, simulate a binary 
    symmetric channel which flips each data bit with probability p

    data : np.ndarray
      1-dimensional array containing a stream of bits
    p : float
      probability by which each bit is flipped

    return : np.ndarray
      data with a number of bits flipped
    """
    flipped_data = np.copy(data)
    for pos in range(len(data)):
          if rn.rand() <= p:
              flipped_data = flip_bit(flipped_data, pos)

    return flipped_data


def decoder_accuracy(m : int, p : float) -> float:
    """
    Given a number of parity bits and a noise level p, simulate the 
    accuracy of a Hamming code with m parity bits over a binary 
    symmetric channel with noise level p.
    Generate 10000 random strings, encode them and send them through 
    the channel. Decode them and check how many codewords are correctly 
    decoded over the total number of transmitted codewords.

    m : int
      The number of parity bits in the Hamming code being tested
    p : float
      The probability of each bit being flipped

    return : float
      The probability of messages being correctly decoded with this
      Hamming code, using the noisy channel of probability p
    """
    # set to 10000 for plots
    num_of_codewords = correct_decodings = 1000 # random codewords
    codewords = create_random_codewords(num_of_codewords, m)
    for code in codewords:
      encoded_code = hamming_encode(code, m)
      noisy_code = binary_symmetric_channel(encoded_code, p) # send through noisy channel
      decoded_code = hamming_decode(noisy_code, m)
      if not np.array_equal(code, decoded_code):
        correct_decodings -= 1 # track number of correctly decoded

    return correct_decodings/num_of_codewords


### Below are helper functions added in addition
### to the required functions for the coursework

# Additional function to help with encoding
def find_parity_bits(data : np.ndarray):
    """ 
    Given a block of data of size k, find the correct number of 
    parity bits needed to encode the data with a Hamming Code.

    data : np.ndarray
      array of shape (k,) with the block of bits to encode

    return : int
      number of parity bits needed for the given data block
    """
    m = 0
    k = len(data)
    while k > 2**m-m-1:
      m += 1
    return m


# Additional function to help with bite flipping
def flip_bit(data : np.ndarray, pos : int) -> np.ndarray:
    """ 
    Flip one bit of some given data in a given position. If no 
    position is given (equal to -1), a random index is choosen.

    data : np.ndarray
      array of shape (n,) with the encoded block of bits

    pos : int
      The index position of the bite to flip.

    return : np.ndarray
      data with one bit flipped
    """
    assert (-1 <= pos < data.size), f"Position {pos} is out of range for data of length {len(data)}"

    if pos == -1:
      pos = rn.randint(0, data.size)
    data[pos] = int(not data[pos])
    
    return data


# Additional function for string to numpy conversion
def string_to_numpy(s : str):
    """
    Turn a given string of 0 and 1 into a numpy ndarray.

    s : str
      string sequence of 0 and 1
    
    return : np.ndarray
      numpy array of 0 and 1 equivalent to s
    """
    np_s = np.ndarray(len(s))
    for i in range(len(s)):
      np_s[i]=(int(s[i]))
    return np_s


# Additional function for random codeword generation
def generate_binary_strings(k : int) -> list:
    """
    Generate all possible binary strings of a given length.

    k : int
      number of bits in each string

    return : list
      list of generated binary strings
    """
    codewords = []
    for i in range(2**k):
      bin = format(i, '0' + str(k) + 'b')
      codewords.append(string_to_numpy(bin))

    return codewords


# Additional function for block code codeword generation
def get_block_codewords(n : int, k : int) -> list:
    """
    Generate all possible codewords for a Hamming Code
    with given codeword length and number of data bits.

    n : int
      length of each codeword
    k : int
      number of data bits 

    return : list
      list of all possible codewords for this code block
    """
    m = n - k
    binary_strings = generate_binary_strings(k)
    block_code = []
    for s in binary_strings:
        block_code.append(hamming_encode(s,m))

    return block_code
   

# Additional function to help with text decoding
def is_valid_text(decoded_string : str) -> bool:
    """ 
    Check if a given string is a valid English text.

    decoded_string : str
      The decoded string to check
    """
    alphabet_chars = [c[0] for c in alphabet]

    if len(decoded_string) < 1:
      return False

    test_word = decoded_string[:10]
    for c in test_word:
      c = str(c)
      if c not in alphabet_chars[26:]: # check if character is a digit
        if c.lower() not in alphabet_chars[:26]: # check for upper and lower case letters
          return False
        
    return True


# Additional function for radnom codeword generation
def create_random_codewords(num : int, m : int) -> list:
    """
    Generate a given number of random codewords suitable for
    a Hamming Code with a given number of parity bits.

    num : int
      number of codewords to generate
    m : int
      number of parity bits to use in the Hamming code (needed
      to calculate codeword lenght)

    return : list
      list of generated codewords
    """
    k = 2**m-m-1
    codewords = [rn.randint(0,2,k) for _ in range(num)]
    return codewords