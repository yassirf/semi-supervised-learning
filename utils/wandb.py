import wandb

def set_up_wandb(args):
    if getattr(args, 'wandb', False):
        *_, data_set, model_name = args.checkpoint.split('/')
        arch, _ = model_name.split('-v')
        
        wandb.init(project=f'proxy_uncertainty_{data_set}',
                   name=model_name,
                   group=arch,
                   dir=args.checkpoint)