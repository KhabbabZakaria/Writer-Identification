import torch
import torch.nn as nn
import torch.nn.functional as F

class GMP(nn.Module):
    """ Generalized Max Pooling
    """
    def __init__(self, lamb):
        super().__init__()
        self.lamb = nn.Parameter(lamb * torch.ones(1)).cuda()
        #self.inv_lamb = nn.Parameter((1./lamb) * torch.ones(1))

    def forward(self, x):
        B, D, H, W = x.shape
        N = H * W
        identity = torch.eye(N).cuda()
        # reshape x, s.t. we can use the gmp formulation as a global pooling operation
        x = x.view(B, D, N)
        x = x.permute(0, 2, 1)
        # compute the linear kernel
        K = torch.bmm(x, x.permute(0, 2, 1)).cuda()
        # solve the linear system (K + lambda * I) * alpha = ones
        A = K + self.lamb * identity
        o = torch.ones(B, N, 1).cuda()
        #alphas, _ = torch.gesv(o, A) # tested using pytorch 1.0.1
        alphas, _ = torch.solve(o, A) # tested using pytorch 1.2.0
        alphas = alphas.view(B, 1, -1)
        xi = torch.bmm(alphas, x)
        xi = xi.view(B, -1)
        # L2 normalization
        xi = nn.functional.normalize(xi)
        return xi

dgmp = GMP(1000)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        x = self.gem(x, p=self.p, eps=self.eps)
        b, c, h, w = x.shape
        return torch.reshape(x, (b, c*h*w))

    def gem(self, x, p=3, eps=1e-6):
        x = x.cuda()
        p = p.cuda()
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p).cuda()

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'
gmp = GeM()

'''sample = torch.rand([5, 512, 32, 32]).cuda()
output1 = dgmp(sample.cuda())
output2 = gmp(sample.cuda())

print(output1.shape, output2.shape)'''