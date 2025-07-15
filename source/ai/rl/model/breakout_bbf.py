import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np


class MLP(nn.Module):
    def __init__(self, in_hiddens=512, med_hiddens=512, out_hiddens=512, layers=1,
                  act=nn.ReLU(), out_act=nn.Identity(), bias=True):
        super().__init__()

        modules=[]
        _hiddens = in_hiddens
        for l in range(layers):
            if l == layers-1:
                act = out_act
                med_hiddens = out_hiddens

            modules.append(nn.Linear(_hiddens, med_hiddens, bias=bias))
            modules.append(act)
            _hiddens = med_hiddens

        self.mlp=nn.Sequential(*modules)

    def forward(self,X):
        return self.mlp(X)


class DQN_Conv(nn.Module):
    def __init__(self, in_hiddens, hiddens, ks, stride, padding=0, max_pool=False,
                 norm=True, act=nn.ReLU(), bias=True, layers=1):
        super().__init__()

        modules = []
        _hiddens = in_hiddens
        for l in range(layers):
            modules.append(nn.Conv2d(_hiddens, hiddens, ks, stride, padding, bias=bias))
            modules.append(nn.BatchNorm2d(hiddens, eps=1e-6) if norm else nn.Identity())
            modules.append(act)
            if l == layers-1:
                modules.append(nn.MaxPool2d(3,2,padding=1) if max_pool else nn.Identity())
            _hiddens = hiddens

        self.conv = nn.Sequential(*modules)
        
    def forward(self, X):
        return self.conv(X)


class BBF_Model(nn.Module):
    def __init__(self, n_actions, hiddens=2048, scale_width=4,
                 num_buckets=51, Vmin=-10, Vmax=10, resize=(96, 72)):
        super().__init__()
        self.support = torch.linspace(Vmin, Vmax, num_buckets).cuda()
        
        self.n_actions = n_actions
        self.hiddens=hiddens
        self.scale_width=scale_width
        self.num_buckets = num_buckets

        self.act = nn.ReLU()
        self.transforms = transforms.Compose([transforms.Resize(resize)])
        
        # self.encoder_cnn = IMPALA_Resnet(scale_width=scale_width, norm=False, init=init_xavier, act=self.act)
        self.encoder_cnn = nn.Sequential(
            DQN_Conv(12, 32*scale_width, 3, 1, 1, norm=True, max_pool=True, act=self.act, layers=1),
            DQN_Conv(32*scale_width, 32*scale_width, 3, 1, 1, norm=True, max_pool=True, act=self.act, layers=3),
            DQN_Conv(32*scale_width, 32*scale_width, 3, 1, 1, norm=True, max_pool=True, act=self.act, layers=3),
            DQN_Conv(32*scale_width, 32*scale_width, 3, 1, 1, norm=False, act=self.act, layers=2)
        )
    
        # Single layer dense that maps the flattened encoded representation into hiddens.
        self.projection = MLP(13824, med_hiddens=hiddens, out_hiddens=hiddens, layers=1)
        self.prediction = MLP(hiddens, out_hiddens=hiddens, layers=1)
                                              
        self.transition = nn.Sequential(DQN_Conv(32*scale_width+n_actions, 32*scale_width, 3, 1, 1, norm=False, act=self.act),
                                        DQN_Conv(32*scale_width, 32*scale_width, 3, 1, 1, norm=False, act=self.act))

        # Single layer dense that maps hiddens into the output dim according to:
        # 1. https://arxiv.org/pdf/1707.06887.pdf -- Distributional Reinforcement Learning
        # 2. https://arxiv.org/pdf/1511.06581.pdf -- Dueling DQN
        self.a = MLP(hiddens, out_hiddens=n_actions*num_buckets, layers=1)
        self.v = MLP(hiddens, out_hiddens=num_buckets, layers=1)

    def forward(self, X, y_action):
        X, z = self.encode(X)
        
        q, action = self.q_head(X)
        z_pred = self.get_transition(z, y_action)

        return q, action, X[:,1:].clone().detach(), z_pred

    def renormalize(self, tensor):
        shape = tensor.shape
        tensor = tensor.view(tensor.shape[0], -1)
        max_value,_ = torch.max(tensor, -1, keepdim=True)
        min_value,_ = torch.min(tensor, -1, keepdim=True)
        return ((tensor - min_value) / (max_value - min_value + 1e-5)).view(shape)

    def preprocess(self, state):
        state=torch.tensor(state, dtype=torch.float, device='cuda') / 255
        state=self.transforms(state.permute(2,0,1))
        return state
    
    def env_step(self, X):
        with torch.no_grad():
            X, _ = self.encode(X)
            _, action = self.q_head(X)
            
            return action.detach()

    def predict(self, X):
        return self.env_step(X)[0][0]

    def encode(self, X):
        batch, seq = X.shape[:2]
        self.batch = batch
        self.seq = seq
        X = self.encoder_cnn(X.contiguous().view(self.batch*self.seq, *(X.shape[2:]))).contiguous()
        X = self.renormalize(X).contiguous().view(self.batch, self.seq, *X.shape[-3:])
        X = X.contiguous().view(self.batch, self.seq, *X.shape[-3:])
        z = X.clone()
        X = X.flatten(-3,-1)
        X = self.projection(X)
        return X, z

    def get_transition(self, z, action):
        z = z.contiguous().view(-1, *z.shape[-3:])
        
        action = F.one_hot(action.clone(), self.n_actions).view(-1, self.n_actions)
        action = action.view(-1, 5, self.n_actions, 1, 1).expand(-1, 5, self.n_actions, *z.shape[-2:])

        z_pred = torch.cat( (z, action[:,0]), 1)
        z_pred = self.transition(z_pred)
        z_pred = self.renormalize(z_pred)
        
        z_preds=[z_pred.clone()]
        

        for k in range(4):
            z_pred = torch.cat( (z_pred, action[:,k+1]), 1)
            z_pred = self.transition(z_pred)
            z_pred = self.renormalize(z_pred)
            
            z_preds.append(z_pred)
        
        
        z_pred = torch.stack(z_preds,1)

        z_pred = self.projection(z_pred.flatten(-3,-1)).view(self.batch,5,-1)
        z_pred = self.prediction(z_pred)
        
        return z_pred

    def q_head(self, X):
        q = self.dueling_dqn(X)
        action = (q*self.support).sum(-1).argmax(-1)
        
        return q, action

    def get_max_action(self, X):
        with torch.no_grad():
            X, _ = self.encode(X)
            q = self.dueling_dqn(X)
            
            action = (q*self.support).sum(-1).argmax(-1)
            return action

    def evaluate(self, X, action):
        with torch.no_grad():
            X, _ = self.encode(X)
            
            q = self.dueling_dqn(X)
            
            action = action[:,:,None,None].expand_as(q)[:,:,0][:,:,None]
            q = q.gather(-2,action)
            
            return q

    def dueling_dqn(self, X):
        X = F.relu(X)
        
        a = self.a(X).view(self.batch, -1, self.n_actions, self.num_buckets)
        v = self.v(X).view(self.batch, -1, 1, self.num_buckets)
        
        q = v + a - a.mean(-2,keepdim=True)
        q = F.softmax(q,-1)
        
        return q
    
    def network_ema(self, rand_network, target_network, alpha=0.5):
        for param, param_target in zip(rand_network.parameters(), target_network.parameters()):
            param_target.data = alpha * param_target.data + (1 - alpha) * param.data.clone()

    def hard_reset(self, random_model=None, alpha=0.5):
        if random_model == None:
            random_model = BBF_Model(self.n_actions, self.hiddens, self.scale_width, self.num_buckets).cuda()

        with torch.no_grad():
            
            self.network_ema(random_model.encoder_cnn, self.encoder_cnn, alpha)
            self.network_ema(random_model.transition, self.transition, alpha)

            self.network_ema(random_model.projection, self.projection, 0)
            self.network_ema(random_model.prediction, self.prediction, 0)

            self.network_ema(random_model.a, self.a, 0)
            self.network_ema(random_model.v, self.v, 0)

    def load(self, load_path):
        self.load_state_dict(torch.load(load_path)['model_state_dict'])