import torch.optim as optim


def checkout_optimizer(args, params):
    if args.optimizer == 'sgd':
        return optim.SGD(
            params,
            lr=args.lr,
            weight_decay=args.wd,
            momentum=args.momentum,
            nesterov=args.nesterov
        )
    else:
        return optim.Adam(
            params,
            lr=args.lr,
            betas=args.momentum,
            weight_decay=args.wd
        )
