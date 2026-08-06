class MissingMAEModel:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "MAE model is not reconstructed yet. "
            "Download/provide mae_pretrain_vit_base.pth and replace utils.models_mae "
            "with the MAE implementation used by the paper."
        )


def mae_vit_base_patch16(*args, **kwargs):
    return MissingMAEModel(*args, **kwargs)
