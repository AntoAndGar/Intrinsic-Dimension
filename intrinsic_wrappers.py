import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# the code is from https://github.com/jgamper/intrinsic-dimensionality with minor fixes


class DenseWrap(nn.Module):
    def __init__(self, module, intrinsic_dimension, device):
        """
        Wrapper to estimate the intrinsic dimensionality of the
        objective landscape for a specific task given a specific model
        :param module: pytorch nn.Module
        :param intrinsic_dimension: dimensionality within which we search for solution
        :param device: cuda device id
        """
        super(DenseWrap, self).__init__()

        # Hide this from inspection by get_parameters()
        self.m = [module]

        self.name_base_localname = []

        # Stores the initial value: \theta_{0}^{D}
        self.initial_value = dict()

        # Stores the randomly generated projection matrix P
        self.random_matrix = dict()

        # Parameter vector that is updated, initialised with zeros as per text: \theta^{d}
        V = nn.Parameter(
            torch.zeros((intrinsic_dimension, 1)).to(device, non_blocking=True)
        )
        self.register_parameter("V", V)
        v_size = (intrinsic_dimension,)

        # Iterates over layers in the Neural Network
        for name, param in module.named_parameters():
            # If the parameter requires gradient update
            if param.requires_grad:

                # Saves the initial values of the initialised parameters from param.data and sets them to no grad.
                # (initial values are the 'origin' of the search)
                self.initial_value[name] = v0 = (
                    param.clone()
                    .detach()
                    .requires_grad_(False)
                    .to(device, non_blocking=True)
                )

                # If v0.size() is [4, 3], then below operation makes it [4, 3, v_size]
                matrix_size = v0.size() + v_size

                # Generates random projection matrices P, sets them to no grad
                self.random_matrix[name] = (
                    torch.randn(matrix_size, requires_grad=False).to(
                        device, non_blocking=True
                    )
                    / intrinsic_dimension**0.5
                )

                # NOTE!: lines below are not clear!
                base, localname = module, name
                while "." in localname:
                    prefix, localname = localname.split(".", 1)
                    base = base.__getattr__(prefix)
                self.name_base_localname.append((name, base, localname))

        for name, base, localname in self.name_base_localname:
            delattr(base, localname)

    def forward(self, x):
        # Iterate over the layers
        for name, base, localname in self.name_base_localname:

            # Product between matrix P and \theta^{d}
            ray = torch.matmul(self.random_matrix[name], self.V)

            # Add the \theta_{0}^{D} to P \dot \theta^{d}
            param = self.initial_value[name] + torch.squeeze(ray, -1)

            setattr(base, localname, param)

        # Pass through the model, by getting the module from a list self.m
        module = self.m[0]
        x = module(x)
        return x


def fastfood_vars(DD, device=0):
    """
    Returns parameters for fast food transform
    :param DD: desired dimension
    :return:
    """
    ll = int(np.ceil(np.log(DD) / np.log(2)))
    LL = 2**ll

    # Binary scaling matrix where $B_{i,i} \in \{\pm 1 \}$ drawn iid
    BB = torch.FloatTensor(LL).uniform_(0, 2).type(torch.LongTensor)
    BB = (BB * 2 - 1).type(torch.FloatTensor).to(device, non_blocking=True)
    BB.requires_grad = False

    # Random permutation matrix
    Pi = torch.LongTensor(np.random.permutation(LL)).to(device, non_blocking=True)
    Pi.requires_grad = False

    # Gaussian scaling matrix, whose elements $G_{i,i} \sim \mathcal{N}(0, 1)$
    GG = (
        torch.FloatTensor(
            LL,
        )
        .normal_()
        .to(device, non_blocking=True)
    )
    GG.to(device, non_blocking=True)
    GG.requires_grad = False

    divisor = torch.sqrt(LL * torch.sum(torch.pow(GG, 2)))

    return [BB, Pi, GG, divisor, LL]


def fast_walsh_hadamard_torched(x, axis=0, normalize=False):
    """
    Performs fast Walsh Hadamard transform
    :param x:
    :param axis:
    :param normalize:
    :return:
    """
    orig_shape = x.size()
    assert axis >= 0 and axis < len(
        orig_shape
    ), "For a vector of shape %s, axis must be in [0, %d] but it is %d" % (
        orig_shape,
        len(orig_shape) - 1,
        axis,
    )
    h_dim = orig_shape[axis]
    h_dim_exp = int(round(np.log(h_dim) / np.log(2)))
    assert h_dim == 2**h_dim_exp, (
        "hadamard can only be computed over axis with size that is a power of two, but"
        " chosen axis %d has size %d" % (axis, h_dim)
    )

    working_shape_pre = [int(np.prod(orig_shape[:axis]))]  # prod of empty array is 1 :)
    working_shape_post = [
        int(np.prod(orig_shape[axis + 1 :]))
    ]  # prod of empty array is 1 :)
    working_shape_mid = [2] * h_dim_exp
    working_shape = working_shape_pre + working_shape_mid + working_shape_post

    ret = x.view(working_shape)

    for ii in range(h_dim_exp):
        dim = ii + 1
        arrs = torch.chunk(ret, 2, dim=dim)
        assert len(arrs) == 2
        ret = torch.cat((arrs[0] + arrs[1], arrs[0] - arrs[1]), axis=dim)

    if normalize:
        ret = ret / torch.sqrt(float(h_dim))

    ret = ret.view(orig_shape)

    return ret


def fastfood_torched(x, DD, param_list=None, device=0):
    """
    Fastfood transform
    :param x: array of dd dimension
    :param DD: desired dimension
    :return:
    """
    dd = x.size(0)

    if not param_list:

        BB, Pi, GG, divisor, LL = fastfood_vars(DD, device=device)

    else:

        BB, Pi, GG, divisor, LL = param_list

    # Padd x if needed
    dd_pad = F.pad(x, pad=(0, LL - dd), value=0, mode="constant")

    # From left to right HGPiH(BX), where H is Walsh-Hadamard matrix
    mul_1 = torch.mul(BB, dd_pad)
    # HGPi(HBX)
    mul_2 = fast_walsh_hadamard_torched(mul_1, 0, normalize=False)

    # HG(PiHBX)
    mul_3 = mul_2[Pi]

    # H(GPiHBX)
    mul_4 = torch.mul(mul_3, GG)

    # (HGPiHBX)
    mul_5 = fast_walsh_hadamard_torched(mul_4, 0, normalize=False)

    ret = torch.div(mul_5[:DD], divisor * np.sqrt(float(DD) / LL))

    return ret


class FastfoodWrapper(nn.Module):
    def __init__(self, module, intrinsic_dimension, device):
        """
        Wrapper to estimate the intrinsic dimensionality of the
        objective landscape for a specific task given a specific model using FastFood transform
        :param module: pytorch nn.Module
        :param intrinsic_dimension: dimensionality within which we search for solution
        :param device: cuda device id
        """
        super(FastfoodWrapper, self).__init__()

        # Hide this from inspection by get_parameters()
        self.m = [module]

        self.name_base_localname = []

        # Stores the initial value: \theta_{0}^{D}
        self.initial_value = dict()

        # Fastfood parameters
        self.fastfood_params = {}

        # Parameter vector that is updated
        # Initialised with zeros as per text: \theta^{d}
        V = nn.Parameter(
            torch.zeros((intrinsic_dimension), device=device)
        )  # .to(device))
        self.register_parameter("V", V)
        V.to(device, non_blocking=True)

        # Iterate over layers in the module
        for name, param in module.named_parameters():
            # If param requires grad update
            if param.requires_grad:

                # Saves the initial values of the initialised parameters from param.data and sets them to no grad.
                # (initial values are the 'origin' of the search)
                self.initial_value[name] = v0 = (
                    param.clone()
                    .detach()
                    .requires_grad_(False)
                    .to(device, non_blocking=True)
                )

                # Generate fastfood parameters
                DD = np.prod(v0.size())
                self.fastfood_params[name] = fastfood_vars(DD, device)

                base, localname = module, name
                while "." in localname:
                    prefix, localname = localname.split(".", 1)
                    base = base.__getattr__(prefix)
                self.name_base_localname.append((name, base, localname))

        for name, base, localname in self.name_base_localname:
            delattr(base, localname)

    def forward(self, x):
        # Iterate over layers
        for name, base, localname in self.name_base_localname:

            init_shape = self.initial_value[name].size()
            DD = np.prod(init_shape)

            # Fastfood transform replace dense P
            ray = fastfood_torched(self.V, DD, self.fastfood_params[name]).view(
                init_shape
            )

            param = self.initial_value[name] + ray

            setattr(base, localname, param)

        # Pass through the model, by getting hte module from a list self.m
        module = self.m[0]
        x = module(x)
        return x


class SparseWrap(nn.Module):
    def __init__(self, module, intrinsic_dimension, device):
        """
        Wrapper to estimate the intrinsic dimensionality of the
        objective landscape for a specific task given a specific model
        :param module: pytorch nn.Module
        :param intrinsic_dimension: dimensionality within which we search for solution
        :param device: cuda device id
        """
        super(SparseWrap, self).__init__()

        # Hide this from inspection by get_parameters()
        self.m = [module]

        self.name_base_localname = []

        # Stores the initial value: \theta_{0}^{D}
        self.initial_value = dict()

        # Stores the randomly generated projection matrix P
        self.random_matrix = dict()

        # Parameter vector that is updated, initialised with zeros as per text: \theta^{d}
        V = nn.Parameter(
            torch.zeros((intrinsic_dimension, 1)).to(device, non_blocking=True)
        )
        self.register_parameter("V", V)
        v_size = (intrinsic_dimension,)

        # Iterates over layers in the Neural Network
        for name, param in module.named_parameters():
            # If the parameter requires gradient update
            if param.requires_grad:

                # Saves the initial values of the initialised parameters from param.data and sets them to no grad.
                # (initial values are the 'origin' of the search)
                self.initial_value[name] = v0 = (
                    param.clone()
                    .detach()
                    .requires_grad_(False)
                    .to(device, non_blocking=True)
                )

                # If v0.size() is [4, 3], then below operation makes it [4, 3, v_size]
                matrix_size = v0.size() + v_size

                # Generates random projection matrices P, sets them to no grad
                self.random_matrix[name] = (
                    torch.randn(matrix_size, requires_grad=False)
                    .to_sparse()
                    .to(device, non_blocking=True)
                    / intrinsic_dimension**0.5
                )
                self.random_matrix[name].coalesce()

                base, localname = module, name
                while "." in localname:
                    prefix, localname = localname.split(".", 1)
                    base = base.__getattr__(prefix)
                self.name_base_localname.append((name, base, localname))

        for name, base, localname in self.name_base_localname:
            delattr(base, localname)

    def forward(self, x):
        # Iterate over the layers
        for name, base, localname in self.name_base_localname:

            # Sparse Product between matrix P and \theta^{d}
            if len(self.random_matrix[name].shape) <= 2:
                # sparse mm for bias vectors
                ray = torch.mm(self.random_matrix[name], self.V)
            else:
                # sparse bmm for weight matrices
                ray = torch.bmm(
                    self.random_matrix[name],
                    self.V.broadcast_to(
                        (
                            self.random_matrix[name].shape[0],
                            self.V.shape[0],
                            self.V.shape[-1],
                        )
                    ),
                )

            self.random_matrix[name].coalesce()

            # Add the \theta_{0}^{D} to P \dot \theta^{d}
            # in the sparse case (sparse + dense) is supported but not (dense + sparse)
            param = torch.squeeze(ray, -1) + self.initial_value[name]

            setattr(base, localname, param)

        # Pass through the model, by getting the module from a list self.m
        module = self.m[0]
        x = module(x)
        return x

def rademacher(shape, device=0):
    """Creates a random tensor of shape under the Rademacher distribution (P(x=1) == P(x=-1) == 0.5)"""
    x = torch.empty(shape, device=device, requires_grad=False).random_(0, 2) # Creates random tensor of 0 and 1
    if device == torch.device("cuda"):
        s = torch.cuda.Stream()  # Create a new stream.
        with torch.cuda.stream(s):
            # this op may start execution before normal_() finishes!
            x[x == 0] = -1  # Turn the 0s into -1
    else:
        x[x == 0] = -1  # Turn the 0s into -1
    return x

def fastJL_vars(DD, d, device=0):
    """
    Returns parameters for fast food transform
    :param DD: desired dimension
    :return:
    """
    #epsilon = 0.1
    ll = int(np.ceil(np.log2(DD)))
    LL = int(np.power(2,ll))

    # random reflection given by the diagonal matrix D ∈ R^d×d where Dii are independent Rademacher random variables
    #D = torch.diag(rademacher(LL, device=device)).to(device, non_blocking=True)
    D = rademacher(LL, device=device)
    D.requires_grad = False
    D.to(device, non_blocking=True)

    n = np.log(60000)**2 # n in the paper is the dataset size, but we use the desired dimension of the projection instead
    #k=int(np.ceil(n/epsilon**2)) # k in the paper is the subspace dimension, because in this work we are estimating it our k depends on the projection/subspace dimension 

    # Pij ≡ bijxrij , where bij ∼ Bernoulli(q) and rij ∼ N (0, q^−1) are independent random variables
    q = min(n/d, 1)
    #print(q)
    B = torch.empty(LL, dtype=torch.float32, device=device, requires_grad=False).bernoulli_(1-q)
    #print("B: ", B)
    
    R = torch.empty(LL, dtype=torch.float32, device=device, requires_grad=False).normal_(0,1/q)
    #print("R: ", R)
    if device == torch.device("cuda"):
        s = torch.cuda.Stream()  # Create a new stream.
        with torch.cuda.stream(s):
            PP = torch.mul(B, R)
    else:
        PP = torch.mul(B, R)
        PP.to(device, non_blocking=True)
    PP.requires_grad = False
    #print(PP)
    #print(PP.shape)

    return [D, PP, LL]


def fastJL_torched(x, DD, param_list=None, device=0):
    """
    Fastfood transform
    :param x: array of dd dimension
    :param DD: desired dimension
    :return:
    """
    dd = x.size(0)
    #print("dd: ", dd)

    if not param_list:

        D, PP, LL = fastJL_vars(DD, dd, device=device)

    else:

        D, PP, LL = param_list

    # Padd x if needed
    dd_pad = F.pad(x, pad=(0, LL - dd), value=0, mode="constant")
    dd_pad.to(device, non_blocking=True)

    # From left to right (1/k)PH(Dx), where H is Walsh-Hadamard matrix
    mul_1 = torch.mul(D, dd_pad)
    mul_1.to(device, non_blocking=True)
    #print("mul_1: ", mul_1.shape)

    # (1/k)P(HDx)
    mul_2 = fast_walsh_hadamard_torched(mul_1, 0, normalize=False)
    mul_2.to(device, non_blocking=True)
    #print("mul_2: ", mul_2.shape)

    # (1/k)(PHDx)
    mul_3 = torch.mul(PP, mul_2)#.flatten()
    mul_3.to(device, non_blocking=True)
    #print("PP: ", PP.shape)
    #print("mul_3: ", mul_3.shape)

    ret = 1/dd * mul_3[:DD]
    ret.to(device, non_blocking=True)

    return ret


class FastJLWrapper(nn.Module):
    def __init__(self, module, intrinsic_dimension, device):
        """
        Wrapper to estimate the intrinsic dimensionality of the
        objective landscape for a specific task given a specific model using FastJL transform
        :param module: pytorch nn.Module
        :param intrinsic_dimension: dimensionality within which we search for solution
        :param device: cuda device id
        """
        super(FastJLWrapper, self).__init__()

        # Hide this from inspection by get_parameters()
        self.m = [module]

        self.name_base_localname = []

        # Stores the initial value: \theta_{0}^{D}
        self.initial_value = dict()

        # Fastfood parameters
        self.fastJL_params = {}

        # Parameter vector that is updated
        # Initialised with zeros as per text: \theta^{d}
        V = nn.Parameter(torch.zeros((intrinsic_dimension), device = device))#.to(device))
        self.register_parameter("V", V)
        V.to(device, non_blocking=True)

        # Iterate over layers in the module
        for name, param in module.named_parameters():
            # If param requires grad update
            if param.requires_grad:

                # Saves the initial values of the initialised parameters from param.data and sets them to no grad.
                # (initial values are the 'origin' of the search)
                self.initial_value[name] = v0 = (
                    param.clone().detach().requires_grad_(False).to(device, non_blocking=True)
                )

                # Generate fastJL parameters
                DD = np.prod(v0.size())
                self.fastJL_params[name] = fastJL_vars(DD, V.size(0), device)

                base, localname = module, name
                while "." in localname:
                    prefix, localname = localname.split(".", 1)
                    base = base.__getattr__(prefix)
                self.name_base_localname.append((name, base, localname))

        for name, base, localname in self.name_base_localname:
            delattr(base, localname)

    def forward(self, x):
        # Iterate over layers
        for name, base, localname in self.name_base_localname:

            init_shape = self.initial_value[name].size()
            #print("init_shape: ", init_shape)
            DD = np.prod(init_shape)
            #print("DD: ", DD)

            # FastJL transform replace dense P
            ray = fastJL_torched(self.V, DD, self.fastJL_params[name]).view(
                init_shape
            )

            param = self.initial_value[name] + ray

            setattr(base, localname, param)

        # Pass through the model, by getting hte module from a list self.m
        module = self.m[0]
        x = module(x)
        return x