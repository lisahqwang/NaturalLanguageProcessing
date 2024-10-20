'''
train a feed-forward neural network to predict the transitions of an arc-standard dependency parser. 
The input to this network will be a representation of the current state (including words on the stack and buffer). 
The output will be a transition (shift, left_arc, right_arc), together with a dependency relation label.
Much of the parser code  is provided, but you will need to implement the input representation for the neural net,
 decode the output of the network, and also specify the network architecture and train the model.
'''

'''
extracts two numpy matrices representing input output pairs. Change input representation to the neural network
'''
#! pip install keras
#! pip install tensorflow

from conll_reader import DependencyStructure, conll_reader
from collections import defaultdict
import copy
import sys
import keras
import os
import numpy as np

'''
don't understand why the self.buffer is a reversed list of input sentence 
'''
class State(object):
    def __init__(self, sentence = []):
        self.stack = []
        self.buffer = []
        if sentence: 
            self.buffer = list(reversed(sentence))
        self.deps = set() 
    
    def shift(self):
        self.stack.append(self.buffer.pop())

    def left_arc(self, label):
        self.deps.add((self.buffer[-1], self.stack.pop(),label))

    def right_arc(self, label):
        parent = self.stack.pop()
        self.deps.add( (parent, self.buffer.pop(), label) )
        self.buffer.append(parent)

    def __repr__(self):
        return "{},{},{}".format(self.stack, self.buffer, self.deps)

   
'''
apply_sequence takes the sequence from somewhere and then just applies it onto the sentence
returns the dependency set of (w_i, r, w_j)
'''
def apply_sequence(seq, sentence):
    state = State(sentence)
    for rel, label in seq:
        if rel == "shift":
            state.shift()
        elif rel == "left_arc":
            state.left_arc(label) 
        elif rel == "right_arc":
            state.right_arc(label) 
         
    return state.deps


class RootDummy(object):
    def __init__(self):
        self.head = None
        self.id = 0
        self.deprel = None    
    def __repr__(self):
        return "<ROOT>"

'''
Obtaining oracle transitions and a sequence of input/output examples
given a dependency structure, returns a list of (state, transition) pairs
'''
def get_training_instances(dep_structure):

    deprels = dep_structure.deprels
    
    sorted_nodes = [k for k,v in sorted(deprels.items())]
    state = State(sorted_nodes)
    state.stack.append(0)

    childcount = defaultdict(int)
    for ident,node in deprels.items():
        childcount[node.head] += 1
 
    seq = []
    while state.buffer: 
        if not state.stack:
            seq.append((copy.deepcopy(state),("shift",None)))
            state.shift()
            continue
        if state.stack[-1] == 0:
            stackword = RootDummy() 
        else:
            stackword = deprels[state.stack[-1]]
        bufferword = deprels[state.buffer[-1]]
        if stackword.head == bufferword.id:
            childcount[bufferword.id]-=1
            seq.append((copy.deepcopy(state),("left_arc",stackword.deprel)))
            state.left_arc(stackword.deprel)
        elif bufferword.head == stackword.id and childcount[bufferword.id] == 0:
            childcount[stackword.id]-=1
            seq.append((copy.deepcopy(state),("right_arc",bufferword.deprel)))
            state.right_arc(bufferword.deprel)
        else: 
            seq.append((copy.deepcopy(state),("shift",None)))
            state.shift()
    return seq   

dep_relations = ['tmod', 'vmod', 'csubjpass', 'rcmod', 'ccomp', 'poss', 'parataxis', 'appos', 'dep', 'iobj', 'pobj', 'mwe', 'quantmod', 'acomp', 'number', 'csubj', 'root', 'auxpass', 'prep', 'mark', 'expl', 'cc', 'npadvmod', 'prt', 'nsubj', 'advmod', 'conj', 'advcl', 'punct', 'aux', 'pcomp', 'discourse', 'nsubjpass', 'predet', 'cop', 'possessive', 'nn', 'xcomp', 'preconj', 'num', 'amod', 'dobj', 'neg','dt','det']


class FeatureExtractor(object):
    
    '''
    constructor takes two vocab files as inputs (file objects)
    stores a word-to-index dictionary in the attribute 
    word_vocab and POS-to-index dictionary in the attribute pos_vocab
    '''
    def __init__(self, word_vocab_file, pos_vocab_file):
        self.word_vocab = self.read_vocab(word_vocab_file)        
        self.pos_vocab = self.read_vocab(pos_vocab_file)        
        self.output_labels = self.make_output_labels()

    def make_output_labels(self):
        labels = []
        labels.append(('shift',None))
    
        for rel in dep_relations:
            labels.append(("left_arc",rel))
            labels.append(("right_arc",rel))
        return dict((label, index) for (index,label) in enumerate(labels))

    def read_vocab(self,vocab_file):
        vocab = {}
        for line in vocab_file: 
            word, index_s = line.strip().split()
            index = int(index_s)
            vocab[word] = index
        return vocab     

    '''
    Returns an encoding of the input to the neural network, ie. a single vector
    top-three words on the buffer and the next-three words on the stack
    '''
    def get_input_representation(self, words, pos, state):
        # TODO: Write this method for Part 2
        result = np.array([])
        if len(state.buffer) > 0:
            buffer1 = self.word_vocab[self.check_depr(words[state.buffer[-1]], pos[state.buffer[-1]])]
        else:
            buffer1 = self.word_vocab['<NULL>']

        if len(state.buffer) > 1:
            buffer2 = self.word_vocab[self.check_depr(words[state.buffer[-2]], pos[state.buffer[-2]])]
        else:
            buffer2 = self.word_vocab['<NULL>']

        if len(state.buffer) > 2:
            buffer3 = self.word_vocab[self.check_depr(words[state.buffer[-3]], pos[state.buffer[-3]])]
        else:
            buffer3 = self.word_vocab['<NULL>']

        if len(state.stack) > 0:
            stack1 = self.word_vocab[self.check_depr(words[state.stack[-1]], pos[state.stack[-1]])]
        else:
            stack1 = self.word_vocab['<NULL>']

        if len(state.stack) > 1:
            stack2 = self.word_vocab[self.check_depr(words[state.stack[-2]], pos[state.stack[-2]])]
        else:
            stack2 = self.word_vocab['<NULL>']

        if len(state.stack) > 2:
            stack3 = self.word_vocab[self.check_depr(words[state.stack[-3]], pos[state.stack[-3]])]
        else:
            stack3 = self.word_vocab['<NULL>']

        return ([stack1, stack2, stack3, buffer1, buffer2, buffer3]) #length 6 
    
    def check_depr(self, word, pos):
        if not pos:
            return '<ROOT>'
        elif pos == 'NNP':
            return '<NNP>'
        elif pos == 'CD':
            return '<CD>'
        else:
            if word.lower() in self.word_vocab:
                return word.lower()
            else:
                return '<UNK>'
            
    '''
    take a (transition, label) pair as its parameter and return a one-hot representation 
    of these actions. Returns an output with one-hot vector of length 91. 
    '''
    def get_output_representation(self, output_pair):  
        # TODO: Write this method for Part 2
        result = self.output_labels[output_pair]
        return keras.utils.to_categorical(result, num_classes=91) #1x91 sparse vector

     
    
def get_training_matrices(extractor, in_file):
    inputs = []
    outputs = []
    count = 0 
    for dtree in conll_reader(in_file): 
        words = dtree.words()
        pos = dtree.pos()
        for state, output_pair in get_training_instances(dtree):
            inputs.append(extractor.get_input_representation(words, pos, state))
            outputs.append(extractor.get_output_representation(output_pair))
        if count%100 == 0:
            sys.stdout.write(".")
            sys.stdout.flush()
        count += 1
    sys.stdout.write("\n")
    return np.vstack(inputs),np.vstack(outputs)
       

if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 


    with open(sys.argv[1],'r') as in_file:   

        extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
        print("Starting feature extraction... (each . represents 100 sentences)")
        inputs, outputs = get_training_matrices(extractor,in_file)
        print("Writing output...")
        np.save(sys.argv[2], inputs)
        np.save(sys.argv[3], outputs)


