import wandb

def setup_wandb(args, model):

    # Extract dataset and model name
    *_, dataset, modelname = args.checkpoint.split('/')
    
    # Augment dataset name
    dataset += "-nl={}".format(args.num_labelled)

    # Get architecture and version number
    arch, version = modelname.split('-v')
    
    wandb.init(
        entity = 'mg-speech-group',
        project = 'proxy-uncertainty-{}'.format(dataset),
        name = modelname,
        group = arch,
    )

    wandb.config = {
        "dataset": args.dataset,
        "num-labelled": args.num_labelled,
        "num-validation": args.num_validation,
        "iters": args.iters,
        "train-l-batch": args.train_l_batch,
        "train-l-augment": args.train_l_augment,
        "train-ul-batch": args.train_ul_batch,
        "train-ul-augment": args.train_ul_augment,
        "loss": args.loss,
        "temperature": args.temperature,
        "proxy-weight": args.proxy_weight,
    }