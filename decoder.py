import sys
import copy

import numpy as np
import torch 

from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from extract_training_data import FeatureExtractor, State
from train_model import DependencyModel

class Parser(object):

    def __init__(self, extractor, modelfile):
        self.extractor = extractor

        # Create a new model and load the parameters
        self.model = DependencyModel(len(extractor.word_vocab), len(extractor.output_labels))
        self.model.load_state_dict(torch.load(modelfile))
        sys.stderr.write("Done loading model")

        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        state = State(range(1,len(words)))
        state.stack.append(0)

        # TODO: Write the body of this loop for part 5
        while state.buffer:
            features = self.extractor.get_input_representation(words, pos, state)
            #convert to tensors 
            features = torch.LongTensor(features).unsqueeze(0)
            #may not need to do since CrossEntropy 
            softmaxvec = self.model(features)

            #top of sorted indices
            sorted_indices = torch.argsort(softmaxvec, descending = True)[0]

            for i in range(0, len(sorted_indices)):
                action, label = self.output_labels[sorted_indices[i].item()]
                if self.isLegal(state, action): 
                    if action == "shift": 
                        state.shift()
                    elif action == "left_arc":
                        state.left_arc(label)
                    elif action == "right_arc":
                        state.right_arc(label)
                    break

        result = DependencyStructure()
        for p,c,r in state.deps:
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))
        return result
    
    #helper isLegal function 
    def isLegal(self, state, action):
        if action == "shift":
            return len(state.buffer) >= 2 or len(state.stack) == 0 
        if action == "left_arc" or action == "right_arc":
            if len(state.stack) == 0: 
                return False
        if action == "left_arc" and state.stack[-1] == 0:
            return False
        else:
            return True


if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r')
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1)

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file:
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
