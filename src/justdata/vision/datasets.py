from justdata.core.registry import register_dataset

register_dataset("cifar10", "classification", modality="vision")
register_dataset("cifar100", "classification", modality="vision")
register_dataset("imagenet", "classification", modality="vision")
register_dataset("imagenette", "classification", modality="vision")
register_dataset("stanford_dogs", "classification", modality="vision")
register_dataset("kitti_road", "segmentation", modality="vision")
register_dataset("zenodo:", "classification", modality="vision")

register_dataset(
    "wilds:camelyon17",
    "classification",
    modality="vision",
    preset="wilds:camelyon17",
)
register_dataset(
    "wilds:fmow",
    "classification",
    modality="vision",
    preset="wilds:fmow",
)
register_dataset(
    "wilds:iwildcam",
    "classification",
    modality="vision",
    preset="wilds:iwildcam",
)
register_dataset(
    "wilds:rxrx1",
    "classification",
    modality="vision",
    preset="wilds:rxrx1",
)
