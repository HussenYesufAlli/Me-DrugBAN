CHARPROTSET = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, "F": 7, "I": 8, "H": 9, "K": 10,
    "M": 11, "L": 12, "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, "U": 19,
    "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25,
}
CHARPROTLEN = 25

def integer_encode_protein(sequence, max_length=1200):
    # Integer encode, pad/truncate
    code = [CHARPROTSET.get(aa, 0) for aa in sequence[:max_length]]
    code += [0] * (max_length - len(code))
    return code