import torch


def sinkhorn_loss(M, a, b, lambda_sh, numItermax=5000, stopThr=0.5e-2):

    u = torch.ones_like(a) / a.size()[0]

    K = torch.exp(-M * lambda_sh)
    err = 1                     # Python float — the `while err > stopThr` compare
    cpt = 0                     # then stays on-host and does NOT sync the GPU
    while err > stopThr and cpt < numItermax:
        u = torch.div(a, torch.matmul(K, torch.div(b, torch.matmul(u.t(), K).t())))
        cpt += 1
        if cpt % 20 == 1:
            v = torch.div(b, torch.matmul(K.t(), u))
            u = torch.div(a, torch.matmul(K, v))
            bb = torch.mul(v, torch.matmul(K.t(), u))
            # `.item()` here forces the sync ONLY every 20 iters (when err actually
            # updates) instead of every iter (via the tensor->bool convert in the
            # while condition). Value passed to the compare is exact; bit-identical.
            err = torch.norm(torch.sum(torch.abs(bb - b), dim=0), p=float("inf")).item()

    sinkhorn_divergences = torch.sum(
        torch.mul(u, torch.matmul(torch.mul(K, M), v)), dim=0
    )

    return sinkhorn_divergences
