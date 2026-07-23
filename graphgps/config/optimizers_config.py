from torch_geometric.graphgym.register import register_config


def extended_optim_cfg(cfg):
    """Extend optimizer config group that is first set by GraphGym in
    torch_geometric.graphgym.config.set_cfg
    """

    # Number of batches to accumulate gradients over before updating parameters
    # Requires `custom` training loop, set `train.mode: custom`
    cfg.optim.batch_accumulation = 1

    # ReduceLROnPlateau: Factor by which the learning rate will be reduced
    cfg.optim.reduce_factor = 0.1

    # ReduceLROnPlateau: #epochs without improvement after which LR gets reduced
    cfg.optim.schedule_patience = 10

    # ReduceLROnPlateau: Lower bound on the learning rate
    cfg.optim.min_lr = 0.0

    # For schedulers with warm-up phase, set the warm-up number of epochs
    cfg.optim.num_warmup_epochs = 50

    # cosine_with_warmup: fraction of a cosine wave to traverse. 0.5 (default) decays to 0 by
    # the last epoch; <0.5 decays slower and ends at a higher LR (0.25 -> ends at 50% of peak,
    # 0.3 -> ~35%, 0.4 -> ~10%).
    cfg.optim.num_cycles = 0.5

    # Clip gradient norms while training
    cfg.optim.clip_grad_norm = False


register_config('extended_optim', extended_optim_cfg)
