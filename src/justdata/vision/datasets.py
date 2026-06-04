from justdata.core.registry import register_dataset

register_dataset("cifar10", "classification", modality="vision")
register_dataset("cifar100", "classification", modality="vision")
register_dataset("imagenet", "classification", modality="vision")
register_dataset("imagenette", "classification", modality="vision")
register_dataset("stanford_dogs", "classification", modality="vision")
register_dataset("voc", "object_detection", modality="vision")
register_dataset("voc/2007", "object_detection", modality="vision")
register_dataset("voc/2012", "object_detection", modality="vision")
register_dataset("nyu_depth_v2_mini", "depth_estimation", modality="vision")
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
