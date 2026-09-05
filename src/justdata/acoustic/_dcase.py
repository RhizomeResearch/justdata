from justdata.acoustic.configs import LabelTransformConfig


DCASE_CLASSES = (
    "airport",
    "shopping_mall",
    "metro_station",
    "street_pedestrian",
    "public_square",
    "street_traffic",
    "tram",
    "bus",
    "metro",
    "park",
)


def _dcase_label_transform() -> LabelTransformConfig:
    return LabelTransformConfig(
        mode="index",
        num_classes=len(DCASE_CLASSES),
        class_names=DCASE_CLASSES,
    )
