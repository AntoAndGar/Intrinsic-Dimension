from networks_arch import *
from torchvision.models import resnet18

def count_params(model, msg="Number of parameters: ", verbose=True):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"{msg}{int(num_params)}")
    return num_params

def write_info(file, num_params, network_type, hidden_dim = None, num_layers = None, net_ty_frmt = None):
    with open(file, 'a') as f:
        f.write("\n##############################################")
        f.write(f"\nNumber_of_parameters: {int(num_params)}")
        if network_type == "fcn":
            f.write(f"\nhidden_dim: {hidden_dim}")
            f.write(f"\nnum_layers: {num_layers}")
        else:
            f.write(f"\ntype: {net_ty_frmt}")
    f.close()

def build_model(network_type, input_dim, hidden_dim, output_dim, num_layers, channel_in, result_file, dataset, device, verbose=True):
    model = None
    num_params = 0
    if network_type == "fcn":
        model = FullyConnectedNetwork(input_dim, hidden_dim, output_dim, num_layers)
        num_params = count_params(model, verbose=verbose)

        #save information to file
        write_info(result_file, num_params, network_type, hidden_dim, num_layers)
    elif network_type == "lenet":
        model = LeNet(channel_in, output_dim, dataset)
        num_params = count_params(model, verbose=verbose)

        #save information to file
        write_info(result_file, num_params, network_type, net_ty_frmt="LeNet")
    elif network_type == "untied_lenet":
        model = Untied_LeNet(channel_in, output_dim)
        num_params = count_params(model, verbose=verbose)
        
        #save information to file
        write_info(result_file, num_params, network_type, net_ty_frmt="Untied_LeNet")
    elif network_type == "fc_lenet":
        model = FcLeNet(input_dim, output_dim)
        num_params = count_params(model, verbose=verbose)

        #save information to file
        write_info(result_file, num_params, network_type, net_ty_frmt="FcLeNet")
    elif network_type == "fc_tied_lenet":
        model = FCTied_LeNet(channel_in, output_dim)
        num_params = count_params(model, verbose=verbose)

        #save information to file
        write_info(result_file, num_params, network_type, net_ty_frmt="FCTied_LeNet")
    elif network_type == "resnet":
        model = resnet18(weights=None) # weights=None to avoid loading pretrained weights
        num_params = count_params(model, verbose=verbose)

        #save information to file
        write_info(result_file, num_params, network_type, net_ty_frmt="ResNet18")
    else:
        raise Exception("Name of network architecture not in: [fcn, lenet, untied_lenet, fc_lenet, fc_tied_lenet, resnet]")

    if verbose:
        modules = [module for module in model.modules()]
        # Print Model Summary
        print(modules[0])
    
    model.to(device)
    return model, num_params