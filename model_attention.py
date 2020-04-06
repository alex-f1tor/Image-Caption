import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2]) 
        self.dim = 2048

        # for param in resnet.parameters():
        #     param.requires_grad_(False)
        
        # modules = list(resnet.children())[:-1]
        # self.resnet = nn.Sequential(*modules)
        # self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        #features = features.view(features.size(0), -1)
        #features = self.embed(features)
        return features
    

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

# class DecoderRNN(nn.Module):
#     def __init__(self, embed_size, hidden_size, vocab_size, num_layers=10):
#         super(DecoderRNN, self).__init__()
#         self.embed_size = embed_size
#         self.hidden_dim = hidden_size
#         self.vocab_size = vocab_size

#         self.word_embeddings = nn.Embedding(self.vocab_size, self.embed_size)
#         self.lstm = nn.LSTM(self.embed_size, self.hidden_dim, num_layers, batch_first=True)
#         self.dropout1 = nn.Dropout()
#         self.fc1 = nn.Linear(self.hidden_dim, self.vocab_size) 
#         # self.dropout2 = nn.Dropout()
#         # self.fc2 = nn.Linear(1024, self.vocab_size)
    
#     def init_weights(self):
#         """Initialize weights."""
#         self.word_embeddings.weight.data.uniform_(-0.1, 0.1)
#         self.fc1.weight.data.uniform_(-0.1, 0.1)
#         self.fc1.bias.data.fill_(0)

#     def forward(self, features, captions):
#         #embeds = self.word_embeddings(captions)
#         #print(embeds.shape)
#         #embeds_cat = torch.cat((features.unsqueeze(1), embeds), 1)
#         #print(embeds_cat.shape)
#         #packed = pack_padded_sequence( embeds_cat, [15]*10, batch_first=True)
# #         print('packed shape', packed)
#         embeddings = self.word_embeddings(captions)
#         #print(embeddings.shape)
#         embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
#         hiddens, _ = self.lstm(embeddings)
#         #lstm_out, _ = self.lstm(embeddings, hc)
#         x = self.dropout1(hiddens)
#         x = F.relu(self.fc1(x))
        

#         return  x

class Attention(nn.Module):
    def __init__(self, encoder_dim):
        super(Attention, self).__init__()
        self.U = nn.Linear(512, 512)
        self.W = nn.Linear(encoder_dim, 512)
        self.v = nn.Linear(512, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(1)

    def forward(self, img_features, hidden_state):
        U_h = self.U(hidden_state).unsqueeze(1)
        W_s = self.W(img_features)
        att = self.tanh(W_s + U_h)
        e = self.v(att).squeeze(2)
        alpha = self.softmax(e)
        context = (img_features * alpha.unsqueeze(2)).sum(1)
        return context, alpha


class DecoderRNN(nn.Module):
    def __init__(self, encoder_dim, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        #define properties
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.hidden_nn = hidden_size*2
        self.encoder_dim = encoder_dim

        self.init_h = nn.Linear(encoder_dim, self.embed_size)
        self.init_c = nn.Linear(encoder_dim, self.embed_size)
        self.tanh = nn.Tanh()


        self.f_beta = nn.Linear(self.embed_size, self.encoder_dim)
        self.sigmoid = nn.Sigmoid()

        self.deep_output = nn.Linear(self.embed_size, self.vocab_size)
        self.dropout = nn.Dropout()


        self.attention = Attention(self.encoder_dim)
        #using the architecture from the paper
        self.embeddings = nn.Embedding(elfvocab_size, self.embed_size)# converts to vocab size, then specified embed size
        self.lstm = nn.LSTMCell(self.embed_size+ encoder_dim, self.hidden_size) # hidden outputs
        
        self.linear1 = nn.Linear(self.hidden_size, self.hidden_nn) #vector of vocab size
        self.linear2 = nn.Linear(self.hidden_nn, self.vocab_size) #vector of vocab size
        #softmax activation turns element of vocab size to vocab scores
        self.bn1 = nn.BatchNorm1d(self.hidden_nn)
        self.softmax = nn.Softmax(dim=1) #(so every slice along a dim will sum to 1).
    
    def forward(self, features, captions): #for the training phase
        #e.g feature -> (10,512), caption -> (10, 14)
        #get batch size
        batch_size = features.shape[0]
        hidden_state, cell_state = self.get_init_lstm_state(features)
        outputs = torch.empty((batch_size, captions.shape[1], self.vocab_size)).cuda()
        alphas = torch.zeros(batch_size, captions.shape[1], img_features.size(1)).cuda()

        #prev_words = torch.zeros(batch_size, 1).long().cuda()
        #embedding = self.embedding(prev_words)


        #initialize hidden outputs and cell states to zeros
        # hidden_state = 0.001*torch.ones((batch_size, self.hidden_size)).cuda()
        # cell_state = 0.001*torch.ones((batch_size, self.hidden_size)).cuda()
        #defining the outpu.t tensor placeholder for each word
        #e.g (10, 14, 8853) each of 14 will have 8853 scores

        #Now embed the captions
        caption_embed = self.embeddings(captions)
        #pass the caption in word by word
        for step in range(captions.shape[1]): #14, so it recurs based on the sampled caption length
            #for the first time step the input is the feature vector (from image)
            context, alpha = self.attention(features, h)
            gate = self.sigmoid(self.f_beta(h))
            gated_context = gate * context

            lstm_input = torch.cat((caption_embed[:,step,:], gated_context), dim=1)
            h, c = self.lstm(lstm_input, (hidden_state, cell_state))


            # if step == 0:
            #     hidden_state, cell_state = self.lstm(features, (hidden_state, cell_state))
            #     print(cell_state.shape)
            # else: #for the 2nd time step and others
                
            #     # [:,step,:] means embedding for that specific time step
            #     hidden_state, cell_state = self.lstm(caption_embed[:,step-1,:], (hidden_state,cell_state))
            #output of the attention mechcanism
            out = self.bn1(F.relu(self.linear1(hidden_state)))
            out = F.relu(self.linear2(out)) #vocab size for a word
            #build the output tensor by storing the vocab size for each word in a big matrix each time step                     
            outputs[:,step,:] = out
            alphas[:,step,:] = alphas
                                                     
        return outputs


    def get_init_lstm_state(self, img_features):
        avg_features = img_features.mean(dim=1)

        c = self.init_c(avg_features)
        c = self.tanh(c)

        h = self.init_h(avg_features)
        h = self.tanh(h)
        return h, c
    
    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features
        #print('inputs', inputs.shape)
        hidden_state = 0.001*torch.ones(( features.shape[0], self.hidden_size)).cuda()
        cell_state = 0.001*torch.ones((features.shape[0], self.hidden_size)).cuda()
        outputs = torch.empty((features.shape[0], 20, self.vocab_size)).cuda()
        
        
        for i in range(20):
            if i==0:
                hidden_state, cell_state = self.lstm(inputs, (hidden_state, cell_state))
            # hiddens: (batch_size, 1, hidden_size)
            elif i==1: #for the 2nd time step and others
                hidden_state, cell_state = self.lstm(self.embeddings(torch.LongTensor(np.zeros(features.shape[0], dtype=np.int16)).to(device)), (hidden_state, cell_state))

                # [:,step,:] means embedding for that specific time step
            else:
                hidden_state, cell_state = self.lstm(inputs2, (hidden_state, cell_state))
            
            out = self.bn1(F.relu(self.linear1(hidden_state)))
            out = F.relu(self.linear2(out))
            if i>1: predicted = torch.max(out[:,2:], 1)[1]+2
            else: predicted = torch.max(out, 1)[1]
            sampled_ids.append(predicted)
            inputs2 = self.embeddings(predicted)                       # inputs: (batch_size, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return list(sampled_ids.squeeze().cpu().numpy())