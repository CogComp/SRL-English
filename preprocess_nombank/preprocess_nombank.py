import nltk.corpus.reader.nombank as nombank_reader
import nltk.corpus
from nltk.data import PathPointer, FileSystemPathPointer
from nltk.tree import Tree

import os
import copy

'''
This file reads nombank files, cleans trees, and fixes indexing to write into the 
following format:

This part is the sentence. ||| sensenum startindex:endindexinclusive-ARGUMENT ...

If the arguments point to parts of the hyphenated word, the corresponding line will look like:

This part is the sentence. ||| sensenum startindex_hyphennumber:endinclusive_hyphennumber-ARGUMENT ...

The output is intentionally left in terms of spans so that the file can be read by a 
model's dataset reader to be put into the specified format, without losing 
information prematurely. Sentences are now ordered in order of their existence in 
the wsj directories and subdirectories, so that all information about the same 
sentence is next to each other.

The train file has wsj subdirectories: 02,03,04,05,06,07,08,09,10,11,12,13,14,15,16,17,18,19,20,21
The eval file has wsj subdirectories: 00,01,22,24
The test file has wsj subdirectories: 23
'''

root_pathp = FileSystemPathPointer('/shared/celinel/noun_srl/nombank.1.0')
nomfile_pathp = FileSystemPathPointer('/shared/celinel/noun_srl/nombank.1.0/sorted_nombank.1.0')
framefiles = '.*\.xml'
words_pathp = FileSystemPathPointer('/shared/celinel/noun_srl/nombank.1.0/nombank.1.0.words')

reader = nombank_reader.NombankCorpusReader(root=root_pathp, 
                                            nomfile=nomfile_pathp, 
                                            framefiles=framefiles, 
                                            nounsfile=words_pathp)

def parsed_sentences(wsj_dir, fileids):
    reader = nltk.corpus.BracketParseCorpusReader(wsj_dir, fileids)
    return reader.parsed_sents()

def read_tree(wsj_dir, inst):
    '''
    Read the tree corresponding to the instance passed in.
    '''
    tree = parsed_sentences(wsj_dir, inst.fileid)
    tree = tree[inst.sentnum]
    return tree


def clean_tree(tree):
    '''
    This function returns a new tree with all artificial tokens removed.

    new_tree, new_indices = clean_tree(tree)
    tree[tree.leaf_treeposition(#)[:-1]] == new_tree[new_tree.leaf_treeposition(new_indices[#])[:-1]] # True

    '''
    
    new_tree = copy.deepcopy(tree)
    to_remove = []
    new_indices = []
    i = 0
    
    for pos in tree.treepositions('leaves'):
        st = tree[pos[:-1]]
        if st.label() == '-NONE-':
            to_remove.append(pos)
            new_indices.append(-1)
        else:
            new_indices.append(i)
            i += 1

    for pos_remove in reversed(to_remove):
        i = 0
        while i < len(pos_remove):
            if len(tree[pos_remove[:-(i+1)]].leaves()) > 1:
                new_tree.__delitem__(pos_remove[:-i])
                break
            i += 1

    assert len(tree.leaves()) == len(new_indices)
    return new_tree, new_indices 


def process_nombanktreepointer(pointer, label, og_tree, new_tree, arg_tree, new_indices, h_indices, all_leaf_treepos):
    # Keep arguments only if its tokens have not been deleted:
    all_removed = True
    for i in range(len(arg_tree.leaves())):
        if new_indices[pointer.wordnum+i] >= 0: # If it is a list, it was expanded, not removed.
            all_removed = False
            break
    if all_removed:
        return None, None, None

    args_start_idx = new_indices[pointer.wordnum+i]

    # Find the difference in height between original and new position
    d_height = len(og_tree.leaf_treeposition(pointer.wordnum)) - len(og_tree.leaf_treeposition(pointer.wordnum+i))
    # Get proper height of argument in new tree.
    height = pointer.height - d_height
    # If the new height is negative, that means we are now at a word "above" the specified level in the tree, so omit the argument.
    if height < 0:
        return None, None, None

    # If argument is a hyphenated chunk, find the proper sections.
    h_nums = []
    for h_idx in h_indices:
        h_nums.append(int(label[h_idx+2:h_idx+3]))


    new_pointer = nombank_reader.NombankTreePointer(args_start_idx, height)
    # Find start of the span (in case args_start_idx is not actual start of arg span)
    treepos = new_pointer.treepos(new_tree)
    span_start_pos = tuple(list(treepos) + list(new_tree[treepos].leaf_treeposition(0)))
    span_start_idx = all_leaf_treepos.index(span_start_pos)
    args_start_idx = [span_start_idx]

    return new_pointer, args_start_idx, h_nums


def get_arguments_spans(og_tree, tree, inst, new_indices):
    '''
    Extract the arguments out of the cleaned tree, corresponding to the instance. 
    '''
    all_leaf_treepos = tree.treepositions('leaves')
    
    #arguments_as_trees = []
    #starts_of_arg_spans = []
    arguments_labels = []
    arg_pointers = []
    for arg in inst.arguments:
        arg_tree = arg[0].select(og_tree) # Get the argument tree from the original tree.
        h_idx = [i for i in range(len(arg[1])) if arg[1].startswith('-H', i)]  
        
        if isinstance(arg[0], nombank_reader.NombankTreePointer):
            new_pointer, arg_starts, arg_hnums = process_nombanktreepointer(arg[0], arg[1], og_tree, tree, arg_tree, new_indices, h_idx, all_leaf_treepos)
            
            if new_pointer == None:
                continue
            arg_tree = new_pointer.select(tree)
            
            subscripted_argstart = "{0}_{1}".format(arg_starts[0], arg_hnums[0]) if len(arg_hnums)>0 else "{0}".format(arg_starts[0])
            subscripted_argend = "{0}_{1}".format(arg_starts[0]+len(arg_tree.leaves())-1, arg_hnums[-1]) if len(arg_hnums)>0 else "{0}".format(arg_starts[0]+len(arg_tree.leaves())-1)
            ss_pointer = "{0}:{1}".format(subscripted_argstart, subscripted_argend)
        else:
            # Chain and Split pointers are lists of NombankTreePointers
            pieces = [i for i in arg[0].pieces]
            new_pieces = []
            arg_starts = []
            arg_ends = []
            for idx, piece in enumerate(pieces):
                new_pointer, start_of_arg, _ = process_nombanktreepointer(piece, arg[1], og_tree, tree, arg_tree, new_indices, h_idx, all_leaf_treepos)
                if new_pointer == None:
                    continue
                new_pieces.append(new_pointer)
                arg_starts.extend(start_of_arg) # Note: this might be problematic if there are ever Chain or Split pointers with ..-HN-HM.. labels. For all of these, we assume that never happens.
                arg_piece_tree = new_pointer.select(tree)
                arg_ends.append(start_of_arg[0]+len(arg_piece_tree.leaves())-1)
            
            if isinstance(arg[0], nombank_reader.NombankSplitTreePointer):
                separator = ','
                new_pointer = nombank_reader.NombankSplitTreePointer(new_pieces)
            elif isinstance(arg[0], nombank_reader.NombankChainTreePointer):
                separator = '*'
                new_pointer = nombank_reader.NombankChainTreePointer(new_pieces)
            
            ss_pointer = ''
            for (start, end) in zip(arg_starts, arg_ends):
                ss_pointer += "{0}:{1}{2}".format(start, end, separator)
            ss_pointer = ss_pointer[:-1]        
        # arg_tree = new_pointer.select(tree)
        
        # arguments_as_trees.append(arg_tree)
        # starts_of_arg_spans.append(arg_starts) 
        if len(h_idx) > 0:
            label = (arg[1][:h_idx[0]]) + arg[1][h_idx[-1]+3:]
            arguments_labels.append(label)
            ss_pointer += "-{0}".format(label)
        else:
            arguments_labels.append(arg[1])
            ss_pointer += "-{0}".format(arg[1])

        arg_pointers.append(ss_pointer)

    pred_idx = new_indices[inst.wordnum]
    h_index = inst.predid.find('-H')
    if h_index < 0:
        pred_pointer = "{0}:{0}-rel".format(pred_idx)
    else:
        h_num = inst.predid[h_index+2:h_index+3] # Should do error checking in reader in case is a faulty data point.
        pred_pointer = "{0}_{1}:{0}_{1}-rel".format(pred_idx, h_num)

    line = "{0} {1} {2} ||| {3} {4} {5}\n".format(inst.fileid, inst.sentnum, " ".join(tree.leaves()), inst.sensenumber, pred_pointer, " ".join(arg_pointers))

    return line

def process_sentences(instances, sentencetree):
    lines = ""
    new_tree, new_indices = clean_tree(sentencetree)
    for inst in instances:
        line = get_arguments_spans(sentencetree, new_tree, inst, new_indices)
        lines = lines+line
    return lines


data = '/shared/celinel/ALGNLP_2/data'
instances = reader.instances()

train_filepath = 'preprocess_nombank/train.srl'
devel_filepath = 'preprocess_nombank/development.srl'
test_filepath = 'preprocess_nombank/test.srl'

if os.path.exists(train_filepath):
    os.remove(train_filepath)
    print('Overwriting existing file at ', train_filepath)
if os.path.exists(devel_filepath):
    os.remove(devel_filepath)
    print('Overwriting existing file at ', devel_filepath)
if os.path.exists(test_filepath):
    os.remove(test_filepath)
    print('Overwriting existing file at ', test_filepath)

with open(train_filepath, "w") as train:
    with open(devel_filepath, "w") as dev:
        with open(test_filepath, "w") as test:
            instances = reader.instances()
            current_filetrees = []
            last_file = ""
            last_sentnum = -1
            sentence_instances = []
            split_set = "train"
            for idx in range(len(instances)):
                # print(current_filetrees)
                inst = instances[idx]
                if last_file[11:13] == "23":
                    split_set = "test"
                elif last_file[11:13] in {"00", "01", "22", "24"}:
                    split_set = "dev"
                else:
                    split_set = "train"
                if inst.fileid == last_file:
                    if inst.sentnum == last_sentnum:
                        sentence_instances.append(inst)
                        continue
                    else:
                        # print('sentence instances: ', sentence_instances)
                        lines = process_sentences(sentence_instances, current_filetrees[last_sentnum])
                        last_sentnum = inst.sentnum
                        sentence_instances = [inst]
                else:
                    if len(sentence_instances) > 0:
                        lines = process_sentences(sentence_instances, current_filetrees[last_sentnum])
                    else:
                        lines = ""
                    current_filetrees = parsed_sentences(data, inst.fileid)
                    last_file = inst.fileid
                    last_sentnum = inst.sentnum
                    sentence_instances = [inst]
                # Process the data for the last sentence.
                if split_set == "test":
                    test.write(lines)
                elif split_set == "dev":
                    dev.write(lines)
                else:
                    train.write(lines)

            # Process data for last sentence in last file.
            if len(sentence_instances) > 0:
                lines = process_sentences(sentence_instances, current_filetrees[last_sentnum])
                if split_set == "test":
                    test.write(lines)
                elif split_set == "dev":
                    dev.write(lines)
                else:
                    train.write(lines)                

