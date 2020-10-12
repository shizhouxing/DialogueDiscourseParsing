# Reference https://github.com/irit-melodi/irit-stac/blob/ec36fac93d26101ba1014db5540483a182472918/stac/harness/ilp.py

from os import path as fp
import numpy as np
import os, re

NUM_LABELS = 16

def pretty_data(data):
    return '\n'.join(
                ' '.join(str(e) for e in lis)
            for lis in data)
            
def dump_scores_to_dat_files(dialog, att_mat, lab_tsr, prefix='default'):
    n_edus = len(dialog["edus"])
    n_labels = NUM_LABELS
    unrelated = 0
    
    tmpdir = "./tmp"
    format_str = '{0:.2f}'

    att_file = os.path.join(tmpdir, '{0}.attach.dat'.format(prefix))
    lab_file = os.path.join(tmpdir, '{0}.label.dat'.format(prefix))

    with open(att_file, 'w') as f:
        f.write('\n'.join(
                    ':'.join(format_str.format(p)
                    for p in row)
                for row in att_mat))
        f.write('\n')

    with open(lab_file, 'w') as f:
        f.write('\n'.join(
                    ' '.join(
                        ':'.join(format_str.format(p)
                        for p in tube)
                    for tube in row)
                for row in lab_tsr))
        f.write('\n')

def mk_zimpl_input(dialog):
    data_dir = "./tmp"
    
    edus = dialog["edus"]
    
    turn_len = []   # Turn lengths
    turn_off = []   # Turn offsets
    edu_ind = []    # Turn indexes for EDUs
    c_off = 0
    
    for i, edu in enumerate(dialog["edus"]):
        edu_ind.append(edu["turn"] + 1)
        
    i = 0
    while i < len(edus):
        j = i
        while j < len(edus) and edus[i]["turn"] == edus[j]["turn"]:
            j += 1
        turn_len.append(j - i)
        turn_off.append(c_off)
        c_off += j - i
        i = j
        
    data_path = fp.join(data_dir, 'turn.dat')
    with open(data_path, 'w') as f_data:
        f_data.write(pretty_data([turn_len, turn_off, edu_ind]) + "\n")

    # Create speaker information
    speakers = {}
    for edu in edus:
        speakers[edu["speaker"]] = len(speakers)
    
    last_mat = np.zeros((len(edus), len(edus)), dtype=int)
    current_last = {}
    for i, edu in enumerate(edus):
        for plast in current_last.values():
            last_mat[plast][i] = 1;
        try:
            current_last[edu["speaker"]] = i
        except KeyError:
            pass
            
    data_path = fp.join(data_dir, 'mlast.dat')
    with open(data_path, 'w') as f_data:
        f_data.write(pretty_data(last_mat) + "\n")

    # class indices that correspond to subordinating relations ;
    # required for the ILP formulation of the Right Frontier Constraint
    # in SCIP/ZIMPL
    
    #subord_idc = [i for i, lbl in enumerate(dpack.labels, start=1)
    #              if lbl in set(SUBORDINATING_RELATIONS)]
    subord_idc = []

    header = '\n'.join((
        "param EDU_COUNT := {0} ;".format(len(edus)),
        "param TURN_COUNT := {0} ;".format(len(turn_off)),
        "param PLAYER_COUNT := {0} ;".format(len(speakers)),
        "param LABEL_COUNT := {0} ;".format(NUM_LABELS),
        "set RSub := {{{0}}} ;".format(
            ', '.join(str(i) for i in subord_idc)),
        "param SUB_LABEL_COUNT := {0} ;".format(len(subord_idc)),
    ))

    template_path = fp.join('template.zpl')
    input_path = fp.join(data_dir, 'input.zpl')

    with open(template_path) as f_template:
        template = f_template.read()

    with open(input_path, 'w') as f_input:
        f_input.write(header + "\n")
        f_input.write(template + "\n")

    return input_path
    
    
def load_scip_output(dialog, output_path):
    def load_pairs():
        r = re.compile('x#(\d+)#(\d+)#(\d+)')
        pairs = []
        labels = []
        t_flag = False
        with open(output_path) as f:
            for line in f:
                m = r.match(line)
                if m:
                    # Start of triplets
                    t_flag = True
                elif t_flag:
                    # End of triplets
                    break
                else:
                    # Not reached triplets yet
                    continue
                si, sj, sr = m.groups()
                pairs.append((int(si) - 1, int(sj) - 1))
                labels.append(int(sr) - 1)
        return zip(*pairs), labels

    # Build map (EDU1, EDU2) -> pair_index
    n_edus = len(dialog["edus"])

    # Build indexes of attached pairs
    
    output_attach, output_labels = load_pairs()
    
    pred = []
    for i in range(len(output_attach[0])):
        pred.append((output_attach[0][i], output_attach[1][i], output_labels[i]))
        
    return pred