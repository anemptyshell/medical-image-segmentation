
def warmup_learning_rate(optimizer, train_steps, warmup_steps, lr, method):
    # gradual warmup_lr
    if warmup_steps and train_steps < warmup_steps:
        warmup_percent_done = train_steps / warmup_steps
        warmup_learning_rate = lr * warmup_percent_done
        learning_rate = warmup_learning_rate
    else:
        # after warm up, decay lr
        for param_group in optimizer.param_groups:
            now_lr = param_group['lr']
        if method == 'sin':
            learning_rate = np.sin(now_lr)
        elif method == 'exp':
            learning_rate = now_lr ** 1.001
    if (train_steps + 1) % 100 == 0:
        print("train_steps:%.3f--warmup_steps:%.3f--learning_rate:%.3f" % (
            train_steps + 1, warmup_steps, learning_rate))
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    return learning_rate