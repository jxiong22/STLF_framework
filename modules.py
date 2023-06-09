import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as tf

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.autograd.set_detect_anomaly(True)

def init_hidden(x, hidden_size: int):

    return torch.zeros(1, x.size(0), hidden_size).to(device)


class FeatureLayer(nn.Module):
    def __init__(self,input_size):
        super(FeatureLayer,self).__init__()
        self.input_size = input_size
        self.attn_layer = nn.Sequential(nn.Linear(self.input_size, self.input_size),
                                      nn.Tanh(),
                                      nn.Linear(self.input_size, self.input_size))

    def forward(self, history_feature, future_feature):
        input_data = torch.cat((history_feature, future_feature),1)
        attn_weights = tf.softmax(self.attn_layer(input_data), 2)

        weighted_input = torch.mul(attn_weights, input_data)
        weighted_history_input=weighted_input[:,:history_feature.shape[1],:]
        weighted_future_input = weighted_input[:, history_feature.shape[1]:, :]
        return attn_weights, weighted_history_input, weighted_future_input


class Encoder(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, T: int):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.T = T
        self.period = int(self.T/24)
        self.num_layers = 1
        self.lstm_layer = nn.LSTM(input_size=self.input_size, hidden_size=hidden_size, num_layers=self.num_layers, bidirectional=True)


    def forward(self, input_data, y_history):
        self.lstm_layer.flatten_parameters()
        input_data = torch.cat((input_data, y_history), dim=2)
        input_encoded, lstm_states = self.lstm_layer(input_data.permute(1,0,2))
        hidden = lstm_states[0]
        cell = lstm_states[1]
        return input_encoded.permute(1,0,2), hidden, cell


class Decoder(nn.Module):

    def __init__(self, hidden_size: int, T: int, f_T:int, m_features, out_feats=1):
        super(Decoder, self).__init__()
        self.input_size = m_features
        self.T = T
        self.f_T = f_T
        self.period = int(self.T/24)
        self.encoder_hidden_size = hidden_size
        self.decoder_hidden_size = hidden_size
        self.attn_layer = nn.Sequential(nn.Linear(2*hidden_size + self.input_size, T),
                                        nn.Tanh(),
                                        nn.Linear(T, T))

        self.lstm_layer = nn.LSTM(input_size=2*hidden_size + self.input_size, hidden_size=hidden_size, bidirectional=True, num_layers=1)
        self.fc = nn.Linear(hidden_size + out_feats, out_feats)

        self.fc_final = nn.Sequential(nn.Linear(2*hidden_size, hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size, out_feats))


    def forward(self, input_encoded, feature_history1, y_history, feature_future1,hidden,cell):
        self.lstm_layer.flatten_parameters()
        feature_history = feature_history1.view(feature_history1.shape[0],self.period,-1,feature_history1.shape[2]).permute(0,1,3,2)
        feature_future = feature_future1.permute(0,2,1).unsqueeze(1)

        hidden_initial = hidden
        cell_initial = cell
        input_encoded = input_encoded.to(device)

        output_final = torch.zeros(input_encoded.size(0), self.f_T, 1).to(device)

        feature_future2 = torch.cat([feature_future] * self.period, 1)
        cos = nn.PairwiseDistance(p=2)

        output = cos(feature_future2.permute(0,3,1,2), feature_history.permute(0,3,1,2))

        dp = torch.sum(output, dim=2)
        dp = torch.pow(torch.rsqrt(dp),2)
        alpha = tf.softmax(dp, dim=1)

        for t in range(self.f_T):

            x = torch.cat((hidden_initial[0].unsqueeze(0),hidden_initial[1].unsqueeze(0), feature_future[:,:,:,t].permute(1,0,2)),dim=2)
            x = tf.softmax(self.attn_layer(x),2).squeeze(0)

            c = torch.mul(x.unsqueeze(2),input_encoded)
            c = c.view(c.shape[0], int(c.shape[1]/24), -1, c.shape[2])
            c = c.sum(2)
            c = torch.mul(c.permute(0,2,1), alpha.reshape(alpha.shape[0],1,alpha.shape[1]))
            c = c.sum(dim=2)
            input_decoder = torch.cat((c, feature_future[:,:,:,t].squeeze(1)), dim=1)

            self.lstm_layer.flatten_parameters()
            out, lstm_output = self.lstm_layer(input_decoder.unsqueeze(0), (hidden_initial, cell_initial))
            hidden_initial = lstm_output[0]
            cell_initial = lstm_output[1]

            output_final1 = self.fc_final(torch.cat((hidden_initial[0],hidden_initial[1]),dim=1)).squeeze(0)
            output_final[:, t, :] = output_final1

        return output_final

