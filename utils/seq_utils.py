# Utilities for working with biological sequences and machine learning

import re

import numpy

def one_hot_encode(sequences, max_seq_len=None, mask_val=0, padding='left'):
    # Dictionary returning one-hot encoding of nucleotides. 
    nuc_d = {'a':[1,0,0,0],
             'c':[0,1,0,0],
             'g':[0,0,1,0],
             't':[0,0,0,1],
             'n':[0,0,0,0]}

    # Automatically use max length if not specified
    if max_seq_len is None:
        max_seq_len = numpy.max([len(s) for s in sequences])
    
    # Creat empty matrix
    one_hot_seqs = numpy.ones([len(sequences), max_seq_len, 4])*mask_val
    
    # Iterate through sequences and one-hot encode
    for i, seq in enumerate(sequences):
        # Truncate
        if padding=='left':
            seq = seq[:max_seq_len]
        elif padding=='right':
            seq = seq[-max_seq_len:]
        else:
            raise ValueError(f'padding {padding} not recognized')
        # Convert to array
        seq = seq.lower()
        one_hot_seq = numpy.array([nuc_d[x] for x in seq])
        # Append to matrix
        if padding=='left':
            one_hot_seqs[i, :len(seq), :] = one_hot_seq
        elif padding=='right':
            one_hot_seqs[i, -len(seq):, :] = one_hot_seq
        else:
            raise ValueError(f'padding {padding} not recognized')
            
    return one_hot_seqs

def load_meme(mofif_filepath):
    """
    Load motif info from meme file

    TODO: Add description of output file structure

    """

    # Background frequencies
    bg_freqs = []

    # Metadata regex
    meta_regex_str = "^letter-probability matrix: alength= (?P<alength>\d+) w= (?P<w>\d+) nsites= (?P<nsites>\d+)$"
    meta_regex = re.compile(meta_regex_str)

    motifs = {}
    with open(mofif_filepath) as f:
        section = 'header'
        while True:
            if section == 'header':
                line = f.readline()
                if not line:
                    break
                # Line with background letter frequencies
                if 'Background letter frequencies' in line:
                    line = f.readline()
                    line_split = line.strip().split()
                    assert(line_split[0]=='A')
                    assert(line_split[2]=='C')
                    assert(line_split[4]=='G')
                    assert(line_split[6]=='T' or line_split[6]=='U')
                    bg_freqs = numpy.array([
                        float(line_split[1]),
                        float(line_split[3]),
                        float(line_split[5]),
                        float(line_split[7]),
                    ])
                # MOTIF marks the start of the motif section
                if line[:5]=='MOTIF':
                    # f.seek(f.tell() - len(line) - 1)
                    readnewline = False
                    section = 'motifs'

            elif section == 'motifs':
                if readnewline:
                    line = f.readline()
                if not line:
                    break
                elif line[:5]=='MOTIF':
                    # New motif
                    line_split = line.strip().split()
                    motif_id = line_split[1]
                    motif_name = line_split[2]
                    # Next line should have matrix metadata
                    line = f.readline()
                    assert(line[:26]=='letter-probability matrix:')
                    m = re.search(meta_regex, line)
                    meta_alength = int(m.group('alength'))
                    meta_w = int(m.group('w'))
                    meta_nsites = int(m.group('nsites'))
                    # meta_E = int(m.group('E'))
                    # Next lines should contain ppm values until line starts with 'URL' or is empty
                    ppm = []
                    line = f.readline()
                    while line[:3]!='URL' and line.strip() != '':
                        ppm.append([float(v) for v in line.strip().split()])
                        line = f.readline()
                        # print(line)
                    ppm = numpy.array(ppm)
                    assert(ppm.shape[0]==meta_w)
                    assert(ppm.shape[1]==meta_alength)
                    # line may contain a url now
                    if line[:3]=='URL':
                        url = line.strip().split()[1]
                    else:
                        url = None

                    # Assemble motif object
                    motif = {
                        'name': motif_name,
                        'ppm': ppm,
                        'url': url,
                        'nsites': meta_nsites,
                        # 'E': meta_E,
                    }
                    motifs[motif_id] = motif
                    readnewline = True
                elif line=='\n':
                    readnewline = True
                else:
                    raise ValueError(f'Unexpected line {line}')

    return motifs


# Save meme files
meme_header = f"""MEME version 4

ALPHABET= ACGU

strands: +

Background letter frequencies
A 0.25 C 0.25 G 0.25 U 0.25

"""

def save_meme_from_dict(meme_dict, filepath, bg_freqs=[0.25, 0.25, 0.25, 0.25]):
    meme_full_str = f"MEME version 4\n\n"
    meme_full_str += f"ALPHABET= ACGU\n\n"
    meme_full_str += f"strands: +\n\n"
    meme_full_str += f"Background letter frequencies\n"
    meme_full_str += f"A {bg_freqs[0]} C {bg_freqs[1]} G {bg_freqs[2]} U {bg_freqs[3]}\n\n"

    for motif_id, motif_info in meme_dict.items():

        meme_motif_full_str = ''
        meme_motif_full_str += f"MOTIF {motif_id} {motif_info['name']}\n"
        meme_motif_full_str += f"letter-probability matrix: alength= 4 w= {len(motif_info['ppm'])} nsites= {motif_info['nsites']}\n"
        for pwm_row in motif_info['ppm']:
            for pwm_val in pwm_row:
                meme_motif_full_str += f'{pwm_val:.6f} '
            meme_motif_full_str += '\n'
        meme_full_str += meme_motif_full_str + '\n'

    with open(filepath, 'w') as file: 
        file.write(meme_full_str)
