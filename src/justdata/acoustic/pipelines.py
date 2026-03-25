from justdata.core.registry import PipelineFuncs, register_pipeline


def _identity_sample(sample, *args, **kwargs):
    return sample


def _identity_batch(batch, num_classes=None, seed=None):
    return batch


@register_pipeline("acoustic/identity")
def identity_pipeline(**kwargs) -> PipelineFuncs:
    return (
        _identity_sample,
        _identity_sample,
        _identity_batch,
        _identity_sample,
    )
