import sys
import numpy as np
import tensorflow as tf
import torch

from torch.nn import Module, Linear, Embedding, NLLLoss, CrossEntropyLoss
from torch.nn.functional import relu, log_softmax
from torch.utils.data import Dataset, DataLoader 

from extract_training_data import FeatureExtractor

class DependencyDataset(Dataset):

  def __init__(self, input_filename, output_filename):
    self.inputs = np.load(input_filename)
    self.outputs = np.load(output_filename)

  def __len__(self): 
    return self.inputs.shape[0]

  def __getitem__(self, k): 
    return (self.inputs[k], self.outputs[k])


class DependencyModel(Module): 

  def __init__(self, word_types, outputs):
    super(DependencyModel, self).__init__()
    # TODO: complete for part 3

    self.embedding = Embedding(word_types, embedding_dim = 128)
    self.linear1 = Linear(in_features = 768, out_features = 128)
    self.linear2 = Linear(in_features = 128, out_features = outputs)

  def forward(self, inputs):
    embedded = self.embedding(inputs)
    #You have to account for the batch size. 
    #When you flatten tensors using view, the first dimension should be left as -1 to pass in inputs of arbitrary batch sizes.

    # x = embedded.view(embedded.size(0), -1)
    # x = relu(self.linear1(x))
    # x = self.linear2(x)

    x = embedded.view(-1, 6*128)
    ReLU = relu(self.linear1(x))
    final_layer = self.linear2(ReLU)
    return final_layer


def train(model, loader): 

  loss_function = CrossEntropyLoss(reduction='mean')

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

    #print("Batch loss: ", loss.item()) # Helpful for debugging, maybe 

    tr_steps += 1
    
    if idx % 1000==0:
      curr_avg_loss = tr_loss / tr_steps
      print(f"Current average loss: {curr_avg_loss}")

    # To compute training accuracy for this epoch 
    correct += sum(torch.argmax(predictions, dim=1) == torch.argmax(targets, dim=1))
    total += len(inputs)
      
    # Run the backward pass to update parameters 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


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
