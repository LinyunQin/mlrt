import torch
from random import shuffle

def gradient_weighting(domain_grads,threshold):
    grad_sum = torch.stack(domain_grads).sum(0)
    grad_sum=torch.nn.functional.normalize(grad_sum, dim=0)
    weights=[]
    for i in range(len(domain_grads)):
        g = domain_grads[i]
        norm2_g = torch.norm(g, p=2, dim=0)
        g = torch.nn.functional.normalize(g, dim=0)
        hat_g = torch.stack(domain_grads[:i]+domain_grads[i+1:]).sum(0)
        norm2_hat_g = torch.norm(hat_g, p=2, dim=0)
        hat_g = torch.nn.functional.normalize(hat_g, dim=0)
        cos_simi=torch.dot(g,grad_sum)
        hat_simi=torch.dot(hat_g,grad_sum)
        if hat_simi-cos_simi>threshold:
            weights.append((norm2_hat_g/norm2_g).unsqueeze_(0))
        else:
            weights.append(torch.ones(1).cuda())
    new_grads = (torch.stack(weights) * torch.stack(domain_grads)).sum(0)
    return new_grads

def get_grads(network):
    grads = []
    for p in network.parameters():
        if p.requires_grad:
            if p.grad is not None:
                #grads.append(torch.tensor(p.grad.data).clone().view(-1).contiguous())
                grads.append(p.grad.data.clone().view(-1).contiguous())
    return torch.cat(grads)


def set_grads(network, new_grads):
    start = 0
    for k, p in enumerate(network.parameters()):
        if p.requires_grad:
            if p.grad is not None:
                dims = p.shape
                end = start + torch.prod(torch.tensor(dims))
                p.grad.data = new_grads[start:end].reshape(dims)
                start = end


def part_get_grads(network):
    grads = []
    for p in network.parameters():
        if p.requires_grad:
            if p.grad is not None:
                #grads.append(torch.tensor(p.grad.data).clone().view(-1).contiguous())
                grads.append(p.grad.data.clone().view(-1).contiguous())
    return torch.cat(grads)


def part_set_grads(network, new_grads):
    start = 0
    for k, p in enumerate(network.parameters()):
        if p.requires_grad:
            if p.grad is not None:
                dims = p.shape
                end = start + torch.prod(torch.tensor(dims))
                p.grad.data = new_grads[start:end].reshape(dims)
                start = end