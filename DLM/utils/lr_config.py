def set_learning_rate(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr


def adjust_lr_poly(optimizer, lr, cur_it, its):
    _lr = lr * (1 - cur_it / its) ** 0.9
    set_learning_rate(optimizer, _lr)