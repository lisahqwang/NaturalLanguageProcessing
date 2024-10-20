# !pip install tensorflow-gpu

import sys
import numpy as np
import torch

from torch.nn import Module, Linear, Embedding, NLLLoss, CrossEntropyLoss
from torch.nn.functional import relu, log_softmax
from torch.utils.data import Dataset, DataLoader 

from extract_training_data import FeatureExtractor

'''
specify and train the neural network, model architecture below and trained weights
'''

class DependencyDataset(Dataset):

  '''
  instantiate the dependencyDataset class and load the data
  from input and output representation 
  '''
  def __init__(self, input_filename, output_filename):
    self.inputs = np.load(input_filename)
    self.outputs = np.load(output_filename)

  '''
  return the total number of input/target pairs in the dataset
  '''
  def __len__(self): 
    return self.inputs.shape[0]

  '''
  return the input/target pair with index k
  '''
  def __getitem__(self, k): 
    return (self.inputs[k], self.outputs[k])


class DependencyModel(Module): 
  '''
  constructor should contain weights which you want to be trained
  embedding layer, 
  '''
  def __init__(self, word_types, outputs):
    super(DependencyModel, self).__init__()
    self.word_types = word_types
    self.outputs = outputs

    self.embedding = torch.nn.Embedding(num_embeddings = word_types, embedding_dim = 128)
    self.linear1 = torch.nn.Linear(in_features = 1, out_features = 128)
    self.linear2 = torch.nn.Linear(in_features = 1, out_features = 91)

  '''
  activation functions like softmax, ReLU, or sigmoid should be in 
  forward function. Also dropout functions are all in forward method 

  Ordinarily we would apply the softmax function to the output to obtain a probability 
  distribution over the 91 different transitions. But the loss function we will use 
  (CrossEntropyLossLinks to an external site.) implicitly computes the softmax for us.
  Instead, the CrossEntropyLoss function  requires the raw logits.
  '''
  def forward(self, inputs):
    embedded = self.embedding(inputs) #passing the input tensor to the embedding layer
    concatenated_embeddings = embedded.view(768) #flattening using .view into a batch size = 768
    ReLU_applied = np.multiple(concatenated_embeddings, torch.nn.functional.relu) # pointwise multiplication, batch size = 91 vector 
    return ReLU_applied

'''
Trains the model for single epoch
'''
def train(model, loader): 

  loss_function = torch.nn.CrossEntropyLoss(reduction='mean')

  LEARNING_RATE = 0.01 
  optimizer = torch.optim.Adagrad(params=model.parameters(), lr=LEARNING_RATE)

  tr_loss = 0 
  tr_steps = 0

  # put model in training mode
  model.train()

  correct = 0 
  total =  0 
  for idx, batch in enumerate(loader):
 
    inputs, targets = batch

    inputs=inputs.type(torch.LongTensor)
    predictions = model(torch.LongTensor(inputs))

    loss = loss_function(predictions, targets)
    tr_loss += loss.item()

    print("Batch loss: ", loss.item()) # Helpful for debugging, maybe 

    tr_steps += 1
    
    if idx % 1000==0:
      curr_avg_loss = tr_loss / tr_steps
      print(f"Current average loss: {curr_avg_loss}")

    # To compute training accuracy for this epoch 
    correct += sum(torch.argmax(predictions, dim=1) == torch.argmax(targets, dim=1))
    total += len(inputs)
      
    # Run the backward pass to update parameters 
    optimizer.zero_grad() # set the gradient for all parameters to 0 
    loss.backward() # recompute gradients based on current loss
    optimizer.step() # update the parameters based on the error gradients 


  epoch_loss = tr_loss / tr_steps
  acc = correct / total
  print(f"Training loss epoch: {epoch_loss},   Accuracy: {acc}")


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


    model = DependencyModel(len(extractor.word_vocab), len(extractor.output_labels))

    dataset = DependencyDataset(sys.argv[1], sys.argv[2])
    loader = DataLoader(dataset, batch_size = 16, shuffle = True)

    print("Done loading data")

    # Now train the model
    for i in range(5): 
      train(model, loader)


    torch.save(model.state_dict(), sys.argv[3]) 
