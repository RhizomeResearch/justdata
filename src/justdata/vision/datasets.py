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
