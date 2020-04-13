"""
file: splinefeatures.py

A collection of networks which compute spline features
of the input. 
"""
import torch
import torch.nn as nn

class HatSpline(nn.Module):
    """
    implements a hat function
    
    INPUTS:
    t -- torch.tensor of size 3
    """
    def __init__(self,t=torch.tensor([0.0,0.5,1.0]),a=0,b=1):
        super(HatSpline,self).__init__()
        self.a = a
        self.b = b
        self.dtype = dtype
        if t.dtype != dtype:
            tt = t.clone().type(dtype)
            self.knot = nn.Parameter(data=tt)
        else:
            self.knot = nn.Parameter(data=t)

        # define the relu
        self.F = nn.functional.relu_

    def forward(self,input):
        # sort the knots first
        knot,idx = torch.sort(self.knot)
        # define coefficients
        c = torch.zeros(3)
        c[0] = 1/(knot[1]-knot[0])
        c[1] = 1/(knot[1]-knot[2])\
                  - 1/(knot[1]-knot[0])
        c[2] = -(c[0]+c[1])
        # evaluate the hat
        out = c[0]*self.F(input - knot[0])\
              + c[1]*self.F(input - knot[1])\
              + c[2]*self.F(input - knot[2])
        return out
    
    def extra_repr(self):
        return 'knot={}'.format(
        self.knot.data
        ) 

    def bound_knots(self):
        """
        ensures that the knots 
        are within the interval of approximation
        """
        with torch.no_grad():
            a = self.a
            b = self.b
            # iterate over knots and move them if they 
            # -- are out of range
            for i in range(3):
                if self.knot.data[i] <= a:
                    self.knot.data[i].uniform_(a+1e-5,a+0.1)
                elif self.knot[i] >= b:
                    self.knot.data[i].uniform_(b-0.1,b-1e-5)
            
class FreeKnotHat(nn.Module):
    r"""
    implementation of a function which computes
    the features assocaited with num_hat free knot
    splines for a single input dimension.
    input is size (num_sample,1)
    \Phi(input) = (\phi_1,\dots,\phi_k)
    where k = num_hat

    PARAMETERS:
    num_hat -- number of hat functions to used
    a       -- left endpoint of interval from which input is sampled
    b       -- right endpoint of interval from which input is sampled

    INPUT: 
    x       -- a torch.tensor sampled from the interval [a,b]
               size should be (num_sample,1)
    
    OUTPUT:
    y       -- a torch.tensor of the spline features assocaited with input x
               size should be (num_sample,num_hat)
    
    NOTES:
    The conv1d layer that evaluates the N_hat hat functions
    has num_hat incoming channels which each channel representing
    the evaluation of a hat function and the length L is 3 (since it 
    takes 3 sigma functions to make the hat function

    self.knot is the collection of the free knots which
    are trainable. each row of self.knot.weight is a triple
    which defines a hat function
    """
    def __init__(self,num_hat=5,a=0,b=1):
        super(FreeKnotHat,self).__init__()
        
        # define number of hats
        self.num_hat = num_hat
        
        # define interval of approximation
        self.a = a
        self.b = b

        # set activation function
        self.F = nn.functional.relu
        
        # initialize knots of the splines
        k = torch.rand(num_hat,3)
        k.uniform_(a,b)
        self.knot = nn.Parameter(data=k)
        
        # convulational layer used to combine F(x-bi)
        self.conv = nn.Conv1d(num_hat,num_hat,3,bias=False,groups=num_hat)
        self.conv.weight.requires_grad = False

    def forward(self,x):
        # set up convulational layer coefficients properly
        k,idx = torch.sort(self.knot,dim=1)
        # compute coefficients
        c1 = 1/(k[:,1:2] - k[:,0:1])
        c2 = 1/(k[:,1:2] - k[:,2:3])\
            -1/(k[:,1:2] - k[:,0:1])
        c3 = -(c1 + c2)

        # set coefficients
        W = torch.cat((c1,c2,c3),1)

        # create F(x-b_i) for i=1,2,3
        y = self.F(x-k.view(3*self.num_hat))
        # y is shape [num_samples,num_hat*3]
        # need to convert it for input into conv
        y = y.view(-1,self.num_hat,3)
        # evaluate all hats
        #y = self.conv(y)
        y = torch.nn.functional.conv1d(y, 
            W.view(self.num_hat,1,3),
            bias=None,
            stride=self.conv.stride[0],
            groups=self.conv.groups)
        # y is shape [num_samples,num_hat,1]
        # output in slightly different shape
        return y.view(x.shape[0],self.num_hat).clamp(min=0)

    def set_hat_coef(self):
        """
        set the coefficeints of the hats based on
        the current values of the knots.
        """
        with torch.no_grad():
            # sort the knots
            k,idx = torch.sort(self.knot,dim=1)
            # compute coefficients
            c1 = 1/(k[:,1:2] - k[:,0:1])
            c2 = 1/(k[:,1:2] - k[:,2:3])\
                -1/(k[:,1:2] - k[:,0:1])
            c3 = -(c1 + c2)

            # set coefficients
            self.W[:,0,0:1] = c1
            self.W[:,0,1:2] = c2
            self.W[:,0,2:3] = c3

    def bound_knots(self):
        """
        ensures that the knots 
        are within the interval of approximation
        """
        with torch.no_grad():
            a = self.a
            b = self.b
            # iterate over knots and move them if they 
            # -- are out of range
            for i,k in enumerate(self.knot):
                for j,val in enumerate(k):
                    """
                    if val <= a:
                        self.knot.data[i,j].uniform_(a+1e-5,a+0.1)
                    elif val >= b:
                        self.knot.data[i,j].uniform_(b-0.1,b-1e-5)
                    """
                    if val < a:
                        self.knot.data[i,j].fill_(a)
                    elif val > b:
                        self.knot.data[i,j].fill_(b)

class prodfks(nn.Module):
    r"""
    prodfks -- Product of free knot spline
    
    Implements the $d$-dimensional function
    $ \Phi(x_1,\dots,x_d) = (\prod_{j=1}^d \phi_{ij}(x_j))_{i=1}^{k}$
    where $k$ is the number of free knot hats and
    $ \phi_{ij}: \mathbb{R} \rightarrow \mathbb{R}$
    is a freeknot hat in one-dimension


    PARAMETERS:
    d       -- dimension of the input
    num_hat -- number of freeknot hats to evaluate for each input
    a       -- left endpoint of hyper-rectangle domain [a,b]^d
    b       -- right endpoint of hyper-rectangle domain [a,b]^d

    INPUT:
    x       -- sampled from hyper rectangle [a,b]^d
               has size (num_sample,d)

    OUTPUT:
    y       -- spline feature evaluation of the input
               has size (num_sample,num_hat)
    """
    def __init__(self,d,num_hat=20,a=0,b=1):
        super(prodfks,self).__init__()
        self.d = d
        self.num_hat = num_hat
        self.a = a
        self.b = b
        
        self.hat = nn.ModuleList()

        # generate spline set for each dimension
        for i in range(d):
            self.hat.append(FreeKnotHat(num_hat,a,b))

    def forward(self,x):
        y = self.hat[0](x[:,0:1])
        for i in range(1,self.d):
            y *= self.hat[i](x[:,i:i+1])
        return y

    def bound_knots(self):
        for i in range(self.d):
            self.hat[i].bound_knots()

class msf(nn.Module):
    r"""
    DOESN'T WORK AS INTENDED
    
    msf - minimum of spline features
    
    Implements the $d$-dimensional function
    $ \Phi(x_1,\dots,x_d) = (\min_{j=1,\dots,d} \phi_{ij}(x_j))_{i=1}^{k}$
    where $k$ is the number of free knot hats and
    $ \phi_{ij}: \mathbb{R} \rightarrow \mathbb{R}$
    is a freeknot hat in one-dimension

    PARAMETERS:
    d       -- dimension of the input
    num_hat -- number of freeknot hats to evaluate for each input
    a       -- left endpoint of hyper-rectangle domain [a,b]^d
    b       -- right endpoint of hyper-rectangle domain [a,b]^d

    INPUT:
    x       -- sampled from hyper rectangle [a,b]^d
               has size (num_sample,d)

    OUTPUT:
    y       -- spline feature evaluation of the input
               has size (num_sample,num_hat)
    """
    def __init__(self,d,num_hat=20,a=0,b=0):
        super(msf,self).__init__()
        self.d = d
        self.num_hat = num_hat
        self.a = a
        self.b = b
        
        self.hat = FreeKnotHat(num_hat,a,b)

    def forward(self,x):
        y = self.hat(x)
        y = y.reshape(-1,self.d,self.num_hat)
        out,_ = torch.min(y,dim=1)
        return out

    def bound_knots(self):
        self.hat.bound_knots()

class minfks(nn.Module):
    r"""
    minfks - minimum of spline features
    
    Implements the $d$-dimensional function
    $ \Phi(x_1,\dots,x_d) = (\min_{j=1,\dots,d} \phi_{ij}(x_j))_{i=1}^{k}$
    where $k$ is the number of free knot hats and
    $ \phi_{ij}: \mathbb{R} \rightarrow \mathbb{R}$
    is a freeknot hat in one-dimension

    PARAMETERS:
    d       -- dimension of the input
    num_hat -- number of freeknot hats to evaluate for each input
    a       -- left endpoint of hyper-rectangle domain [a,b]^d
    b       -- right endpoint of hyper-rectangle domain [a,b]^d

    INPUT:
    x       -- sampled from hyper rectangle [a,b]^d
               has size (num_sample,d)

    OUTPUT:
    y       -- spline feature evaluation of the input
               has size (num_sample,num_hat)
 
    """
    def __init__(self,d,num_hat=20,a=0,b=1):
        super(minfks,self).__init__()
        self.d = d
        self.num_hat = num_hat
        self.a = a
        self.b = b
        
        self.hat = nn.ModuleList()

        # generate spline set for each dimension
        for i in range(d):
            self.hat.append(FreeKnotHat(num_hat,a,b))

    def forward(self,x):
        y = self.hat[0](x[:,0:1])
        for i in range(1,self.d):
            y,_ = torch.min(
                    torch.cat((y,self.hat[i](x[:,i:i+1])),dim=1),
                    dim=1
                  )
            y = y.view(-1,1)
        return y

    def bound_knots(self):
        for i in range(self.d):
            self.hat[i].bound_knots()

