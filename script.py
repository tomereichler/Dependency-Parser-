import matplotlib.pyplot as plt
from chu_liu_edmonds import decode_mst
import torch
import torch.nn as nn
import numpy as np
import time


class EdgeScore_function(nn.Module):
    def __init__(self, in_dim):
        super(EdgeScore_function, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, int(100)),
            nn.Tanh(),
            nn.Linear(int(100), 1)
        )

    def forward(self, encodings):
        number_of_words = encodings.shape[0]
        score_vec = []
        for i in range(number_of_words):
            tensor1 = encodings[i, :].repeat(number_of_words, 1)
            x = torch.cat((tensor1, encodings), 1)
            score = self.layers(x)
            score_vec.append(score)
        score_mat = torch.stack(score_vec).squeeze()
        return score_mat


class KiperwasserDependencyParser(nn.Module):
    def __init__(self, word_dict, word_embedding_dim, hidden_dim, pos_dict, pos_embedding_dim):
        super(KiperwasserDependencyParser, self).__init__()
        self.word_dict = word_dict
        self.pos_dict = pos_dict
        self.word_embedding = nn.Embedding(len(word_dict), word_embedding_dim)
        self.pos_embedding = nn.Embedding(len(pos_dict), pos_embedding_dim)
        self.embedding_dim = self.word_embedding.embedding_dim + self.pos_embedding.embedding_dim
        self.encoder = nn.LSTM(input_size=self.embedding_dim, hidden_size=hidden_dim, num_layers=2, bidirectional=True, batch_first=False)
        self.EdgeScore_function = EdgeScore_function(4*hidden_dim)

    def forward(self, sen, POSs):
        word_tensor = sen
        pos_tensor = POSs
        words_embeddings = self.word_embedding(word_tensor)
        pos_embeddings = self.pos_embedding(pos_tensor)
        final, _ = self.encoder(torch.cat((words_embeddings, pos_embeddings), dim=-1))
        score_mat = self.EdgeScore_function(final)

        return score_mat


# create one hot matrix for heads
def one_hot_representation(h):
    y = torch.zeros(len(h), len(h))
    for x, c in zip(h, range(len(h))):
        y[c, x.to(int)] = 1
    return y


# calculate UAS
def UAS(pred, deps):
    degree = len(deps)
    correct = 0
    for p, d in zip(pred, deps):
        if p == d:
            correct += 1
    return correct / degree


# build a dictionary of words
def preprocessing_dict(lines):
    counter = 0
    pos_counter = 0
    word_dic = {}
    word_count = {}
    pos_dic = {}
    for line in lines:
        if line != '\n':
            row_i = line.split('\t')
            word, pos = row_i[1], row_i[3]
            if pos not in pos_dic:
                pos_dic[pos] = pos_counter
                pos_counter = pos_counter + 1
            if word in word_dic:
                word_count[word] += 1
            else:
                word_dic[word] = counter
                counter = counter + 1
                word_count[word] = 1

    pos_dic["<OOV>"] = pos_counter
    word_dic["<OOV>"] = counter
    return word_dic, pos_dic, word_count


# encode sentence word by word
def sentence_encoding(sen, dic, word_count):
    stack = []
    for word in sen:
        beta = 0.25
        if word in dic:
            if word_count is not None:
                probability = np.random.uniform()
                cutoff = beta / (beta + word_count[word])
                if probability < cutoff:
                    ind = dic["<OOV>"]
                else:
                    ind = dic[word]
            else:
                ind = dic[word]
        else:
            ind = dic["<OOV>"]
        stack.append(torch.tensor(ind).unsqueeze(0))
    return torch.stack(stack).squeeze(1)


# encode list of POS for a sentence
def poses_encoding(POSs, dic):
    stack = []
    for POS in POSs:
        if POS in dic:
            ind = dic[POS]
        else:
            ind = dic["<OOV>"]
        stack.append(torch.tensor(ind).unsqueeze(0))
    return torch.stack(stack).squeeze(1)


# encode POS by POS dictionary
def pos_encoding(POS, dic):
    if POS not in dic:
        ind = dic["<OOV>"]
    else:
        ind = dic[POS]
    return torch.tensor(ind).unsqueeze(0)


# load and encode a file, create dictionary if its a test set
def preprocessing(file_path, device, dic=None, pos_dic=None):
    sen = []
    heads = []
    counters = []
    poss = []
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Check if a dictionary was provided
    if dic is None or pos_dic is None:
        dic, pos_dic, word_counter = preprocessing_dict(lines)
    else:
        word_counter = None

    # encode for all sentences in the file
    sentence = ['ROOT']
    head = [-1]
    counter = [0]
    pos = ['ROOT']
    for line in lines:
        if line == '\n':
            heads.append(torch.Tensor(head))
            head = [-1]
            counters.append(torch.Tensor(counter))
            counter = [0]
            poss.append(poses_encoding(pos, pos_dic))
            pos = ['ROOT']
            sen.append(sentence_encoding(sentence, dic, word_counter))
            sentence = ['ROOT']
        else:
            row = line.split('\t')
            counter.append(int(row[0]))
            sentence.append(row[1])
            h = row[6]
            if h == '_':
                h = '-1'
            head.append(int(h))
            pos.append(row[3])
    return sen, heads, counters, poss, dic, pos_dic


# UAS score and loss
def eval_model(model, sentences, heads, POSs, criterion):
    correct, total = 0, 0
    loss = 0
    preds = np.array([])
    targets = np.array([])
    with torch.no_grad():
        for i, sentence, head, pos in zip(range(len(sentences)), sentences, heads, POSs):
            output = model(sentence, pos)
            y = one_hot_representation(head)
            loss += criterion(output[1:, :], y[1:, :]).item()
            pred, _ = np.array(decode_mst(output.detach().numpy().T, len(head), has_labels=False))
            preds = np.concatenate((preds, pred))
            targets = np.concatenate((targets, head.detach().numpy()))
    uas = UAS(preds, targets)
    final_loss = loss/len(sentences)
    return uas, final_loss


train_path = 'train.labeled'
test_path = 'test.labeled'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Loading Train and Test Sentences
sentences, heads, counters, POSs, word_dic, pos_dic = preprocessing(train_path, device)
test_sentences, test_heads, test_counters, test_POSs, _, _ = preprocessing(test_path, device, word_dic, pos_dic)

# Hyperparameters
word_embedding_dim = 100
pos_embedding_dim = 50
hidden_dim = 150
epochs = 20
lr = 0.005

# _______________Training process______________
# Timing the training
start = time.time()
# Initialize model, optimizer and criterion
model = KiperwasserDependencyParser(word_dic, word_embedding_dim, hidden_dim, pos_dic, pos_embedding_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss().to(device)
# Setting up result lists

loss_train = []
uas_train = []
uas_test = []
loss_test = []
best_uas = 0
gradient_steps = 50
for epoch in range(epochs):
    preds = np.array([])
    targets = np.array([])
    accumulated_loss = 0
    print('Epoch: %d' % (epoch + 1))
    for i, sentence, head, pos in zip(range(len(sentences)), sentences, heads, POSs):
        sentence.to(device)
        output = model(sentence, pos)
        y = one_hot_representation(head)

        # Remove the root from the loss
        loss = criterion(output[1:, :], y[1:, :])
        loss /= gradient_steps
        loss.backward()

        # predictions
        pred, _ = np.array(decode_mst(output.to('cpu').detach().numpy().T, len(head), has_labels=False))
        preds = np.concatenate((preds, pred))
        targets = np.concatenate((targets, head.to('cpu').detach().numpy()))


        if i > 0 and i % gradient_steps == 0:
            optimizer.step()
            model.zero_grad()
        accumulated_loss += loss.item()
        if i > 0 and i % 1000 == 0:
            print('Epoch [%d/%d], i: %d, Loss: %.5f, Train UAS: %s'
                  % (epoch + 1, epochs, i, accumulated_loss/i, str(round(UAS(preds, targets)*100, 3))+'%'))

    uas = UAS(preds, targets)
    uas_score, loss_score = eval_model(model, test_sentences, test_heads, test_POSs, criterion)
    print(f'Epoch {epoch + 1} : Test UAS is {str(uas_score * 100)}%')
    print('*' * 20)
    # Saving Epoch Results
    uas_test.append(uas_score)
    loss_test.append(loss_score)
    loss_train.append(accumulated_loss / len(sentences))
    uas_train.append(uas)
    # Save the Model if epoch's UAS is better than existing best
    if uas_score > best_uas:
        best_uas = uas_score
        best_epoch = epoch + 1
        torch.save(model, 'full_model.pkl')

# Print Training Time
end = time.time()
print(f'Training took {(end - start) / 60} minutes')
print(f'Saved Model\'s Test UAS: {best_uas * 100} Train UAS: {uas_train[best_epoch] * 100} at epoch {best_epoch}')

#loss plot
n = len(loss_train)
epochs = np.arange(n)
plt.plot(epochs, loss_train)
plt.title('Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Results')
plt.show()

#Train UAS plot
n = len(uas_train)
epochs = np.arange(n)
plt.plot(epochs, uas_train)
plt.title('Train UAS')
plt.xlabel('Epochs')
plt.ylabel('Results')
plt.show()

#Train UAS plot
n = len(uas_test)
epochs = np.arange(n)
plt.plot(epochs, uas_test)
plt.title('Test UAS')
plt.xlabel('Epochs')
plt.ylabel('Results')
plt.show()

#Train UAS plot
n = len(uas_train)
epochs = np.arange(n)
plt.plot(epochs, uas_train)
plt.title('Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Results')
plt.show()

