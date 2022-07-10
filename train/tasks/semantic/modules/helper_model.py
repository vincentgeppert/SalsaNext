import torch.nn as nn

class helper_model(nn.Module):
    def __init__(self, nclasses):
        super(helper_model, self).__init__()
        self.nclasses = nclasses

        self.logits = nn.Conv2d(32, nclasses, kernel_size=(1, 1))