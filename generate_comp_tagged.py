from hw3 import *
import numpy as np
from chu_liu_edmonds import decode_mst

test_path_file = 'comp.unlabeled'
pred_path_file = './comp_318736501_208743658.labeled'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = torch.load('model_318736501_208743658.pkl')
word_dict = model.word_dict
pos_dict = model.pos_dict
sentences, heads, counters, poss, _, _ = preprocessing(test_path_file, device, word_dict, pos_dict)

correct, total = 0, 0
targets, predictions = np.array([]) , np.array([])
with torch.no_grad():
    for i, sentence, head, pos in zip(range(len(sentences)), sentences, heads, poss):
        output = model(sentence, pos)
        pred, _ = np.array(decode_mst(output.to('cpu').detach().numpy().T, len(head), has_labels=False))
        predictions = np.concatenate((predictions, pred[1:]))
        targets = np.concatenate((targets, head.to('cpu').detach().numpy()))


test_file = open(test_path_file, 'r')
Lines = test_file.readlines()
output_file = open(pred_path_file, "w")
pred_ind = 0
for line in Lines:
    if line != '\n':
        x = line.split('\t')
        p = predictions[pred_ind]
        x[6] = str(int(p))
        output_file.write('\t'.join(x))
        pred_ind += 1
    else:
        output_file.write("\n")
output_file.close()
