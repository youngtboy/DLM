def set_learning_rate(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr
