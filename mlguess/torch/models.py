from torch import nn
from torch.nn import init
import torch, logging
from torch.nn.utils import spectral_norm as SpectralNorm
import warnings
import random
import numpy as np
import os

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def init_weights(net, init_type='normal', init_gain=0.0, verbose=True):
    """Initialize network weights.
    Parameters:
        net: Network. Network to be initialized.
        init_type: String. The name of an initialization method: normal | xavier | kaiming | orthogonal.
        init_gain: Float. Scaling factor for normal, xavier and orthogonal.
        verbose: Boolean. Verbosity mode.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    if verbose:
        logging.info('Initializing network with %s' % init_type)
    net.apply(init_func)
    
    
class LinearNormalGamma(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = SpectralNorm(nn.Linear(in_channels, out_channels*4))

    def evidence(self, x):
        return  torch.log(torch.exp(x) + 1)

    def forward(self, x):
        pred = self.linear(x).view(x.shape[0], -1, 4)
        mu, logv, logalpha, logbeta = [w.squeeze(-1) for w in torch.split(pred, 1, dim=-1)]
        return mu, self.evidence(logv), self.evidence(logalpha) + 1, self.evidence(logbeta)
    
    
class DNN(torch.nn.Module):
    
    def __init__(self, 
                 input_size,
                 output_size,
                 block_sizes = [1000], 
                 dr = [0.5], 
                 batch_norm = True, 
                 lng = False):
        
        super(DNN, self).__init__()
        self.lng = lng
        
        if len(block_sizes) > 0:
            blocks = self.block(input_size, block_sizes[0], dr[0], batch_norm)
            if len(block_sizes) > 1:
                for i in range(len(block_sizes)-1):
                    blocks += self.block(block_sizes[i], block_sizes[i+1], dr[i], batch_norm)
            if lng:
                blocks.append(LinearNormalGamma(block_sizes[-1], output_size))
            else:
                blocks.append(SpectralNorm(torch.nn.Linear(block_sizes[-1], output_size)))
        else:
            if lng:
                blocks = [LinearNormalGamma(input_size, output_size)]
            else:
                blocks = [SpectralNorm(torch.nn.Linear(input_size, output_size))]
        self.fcn = torch.nn.Sequential(*blocks)
        #self.apply(init_weights)
        
    def block(self, input_size, output_size, dr, batch_norm):
        block = [
            SpectralNorm(torch.nn.Linear(input_size, output_size))
        ]
        if batch_norm:
            block.append(torch.nn.BatchNorm1d(output_size))
        block.append(torch.nn.LeakyReLU())
        if dr > 0.0:
            block.append(torch.nn.Dropout(dr))
        return block

    def forward(self, x):
        x = self.fcn(x)
        return x
    
    def load_weights(self, weights_path: str) -> None:

        """
            Loads model weights given a valid weights path
            
            weights_path: str
            - File path to the weights file (.pt)
        """
        logger.info(f"Loading model weights from {weights_path}")

        try:
            checkpoint = torch.load(
                weights_path,
                map_location=lambda storage, loc: storage
            )
            self.load_state_dict(checkpoint["model_state_dict"])
        except Exception as E:
            logger.info(
                f"Failed to load model weights at {weights_path} due to error {str(E)}"
            )
    
    def predict(self, x, batch_size = 32, return_numpy = True):
        
        if len(x.shape) != 2:
            logger.warning(
                f"The input size should be (batch_size, input size), but received {x.shape}"
            )
            raise
            
        device = get_device()
        logger.info(f"Mounting the model to device {device}")
        self.to(device) 
        self.eval()
        
        if type(x) == torch.Tensor: #
            x = x.numpy()
        
        if self.lng:
            with torch.no_grad():
                if batch_size < x.shape[0]:
                    x_batch = np.array_split(x, x.shape[0] / batch_size)
                    pred = torch.cat([
                        torch.hstack(
                            self.forward(torch.from_numpy(_x).float().to(device))
                        ) for _x in x_batch
                    ]).cpu()
                else:
                    pred = torch.hstack(
                        self.forward(torch.from_numpy(x).float().to(device))
                    ).cpu()
        else:
            with torch.no_grad():
                if batch_size < x.shape[0]:
                    x_batch = np.array_split(x, x.shape[0] / batch_size)
                    pred = torch.cat([
                        self.forward(torch.from_numpy(_x).float().to(device))
                        for _x in x_batch
                    ]).cpu()
                else:
                    pred = self.forward(
                        torch.from_numpy(x).float().to(device)
                    ).cpu()
        return pred.numpy() if return_numpy else pred