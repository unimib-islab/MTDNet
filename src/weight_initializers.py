import torch
import torch.nn as nn
from torch.nn import init


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
    else:
        init.normal_(m.weight.data, 0.0, 0.02)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
    else:
        init.xavier_normal_(m.weight.data, gain=0.02)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
    else:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')


def weights_init_kaiming_small(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        with torch.no_grad():
            m.weight.data.mul_(0.1)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        with torch.no_grad():
            m.weight.data.mul_(0.1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
        with torch.no_grad():
            m.weight.data.mul_(0.1)
    else:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        with torch.no_grad():
            m.weight.data.mul_(0.1)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('LSTM') != -1:

        for value in m.state_dict():
            # format values
            param = m.state_dict()[value]
            if 'weight_ih' in value or 'weight_hh' in value:
                # print(value,param.shape,'Orthogonal')
                # input TO hidden ORTHOGONALLY || Wii, Wif, Wic, Wio
                torch.nn.init.orthogonal_(m.state_dict()[value])
            # elif 'weight_hh' in value:
            #     #INITIALIZE SEPERATELY EVERY MATRIX TO BE THE IDENTITY AND THE STACK THEM
            #     weight_hh_data_ii = torch.eye(self.hidden_units,self.hidden_units)#H_Wii
            #     weight_hh_data_if = torch.eye(self.hidden_units,self.hidden_units)#H_Wif
            #     weight_hh_data_ic = torch.eye(self.hidden_units,self.hidden_units)#H_Wic
            #     weight_hh_data_io = torch.eye(self.hidden_units,self.hidden_units)#H_Wio
            #     weight_hh_data = torch.stack([weight_hh_data_ii,weight_hh_data_if,weight_hh_data_ic,weight_hh_data_io], dim=0)
            #     weight_hh_data = weight_hh_data.view(self.hidden_units*4,self.hidden_units)
            #     #print(value,param.shape,weight_hh_data.shape,self.number_of_layers,self.hidden_units,'Identity')
            #     m.state_dict()[value].data.copy_(weight_hh_data)#hidden TO hidden IDENTITY.state_dict()[value].data.copy_(weight_hh_data)#hidden TO hidden IDENTITY
            # elif 'bias' in value:
            #     # print(value,param.shape,'Zeros')
            #     torch.nn.init.constant_(m.state_dict()[value], val=0)
            #     # set the forget gate | (b_ii|b_if|b_ig|b_io)
            #     m.state_dict()[
            #         value].data[self.hidden_units:self.hidden_units*2].fill_(1)


def init_weights(net, init_type='normal'):
    print('\nNetwork initialization method: %s' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'kaiming_small':
        net.apply(weights_init_kaiming_small)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            'initialization method [%s] is not implemented' % init_type)
