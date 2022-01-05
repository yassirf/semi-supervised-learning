from utils.meter import AverageMeter


def accuracy(output, target, top_k=(1,)):
    """
    Computes the precision@k for the specified values of k
    """
    max_k = max(top_k)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in top_k:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    if len(res) == 1:
        res = res[0]

    return res


class BaseLoss(object):
    def __init__(self, args, model, optimiser, scheduler):
        self.args = args
        self.model = model
        self.optimiser = optimiser
        self.scheduler = scheduler
        self.usegrad = (optimiser is not None)
        self.metrics = {}

    def reset_optimiser(self):
        self.optimiser.zero_grad()

    def reset_metrics(self):
        for key in self.metrics:
            self.metrics[key].reset()

    def record_metrics(self, batch_metrics, batch_size = 1):
        for key, value in batch_metrics.items():
            if key not in self.metrics:
                self.metrics[key] = AverageMeter()
            self.metrics[key].update(value, batch_size)

    def step(self, loss):
        loss.backward()
        self.optimiser.step()
        self.scheduler.step()
        self.lr = self.scheduler.get_last_lr()[0]

    def forward(self, info):
        raise NotImplementedError

    def __call__(self, info, batch_size = 1):

        if self.usegrad:
            # Reset optimiser first
            self.reset_optimiser()

        # Get the loss based on model predictions in info
        loss, linfo = self.forward(info)

        if self.usegrad:
            # Perform a step
            self.step(loss)

        # Record the losses in linfo
        self.record_metrics(linfo['metrics'], batch_size = batch_size)