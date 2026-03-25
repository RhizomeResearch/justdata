from justdata.acoustic.compat.ced import (
    ced_base_16k_logmel64,
    ced_frontend,
    ced_mini_16k_logmel64,
    ced_small_16k_logmel64,
    ced_tiny_16k_logmel64,
    dcase2025_task1_ced_16k_1s,
)
from justdata.acoustic.compat.efficientat import (
    DCASE_CLASSES,
    dcase2025_task1_dymn_32k_1s,
    dcase2025_task1_efficientat_32k_1s,
    dcase2025_task1_efficientat_32k_1s_repeat_to_10s,
    dcase2025_task1_efficientat_32k_1s_zero_pad_to_10s,
    dymn_32k_10s_logmel128,
    efficientat_32k_10s_logmel128,
    efficientat_frontend,
)
from justdata.acoustic.compat.panns import (
    panns_cnn14_16k_10s_logmel64,
    panns_cnn14_32k_10s_logmel64,
    panns_frontend,
)
from justdata.acoustic.compat.passt import (
    PATCH_GRID,
    PATCHOUT,
    dcase2025_task1_passt_32k_1s,
    dcase2025_task1_passt_32k_1s_repeat_to_10s,
    dcase2025_task1_passt_32k_1s_zero_pad_to_10s,
    passt_32k_10s_logmel128,
    passt_frontend,
)

__all__ = [
    "DCASE_CLASSES",
    "PATCH_GRID",
    "PATCHOUT",
    "ced_base_16k_logmel64",
    "ced_frontend",
    "ced_mini_16k_logmel64",
    "ced_small_16k_logmel64",
    "ced_tiny_16k_logmel64",
    "dcase2025_task1_ced_16k_1s",
    "dcase2025_task1_dymn_32k_1s",
    "dcase2025_task1_efficientat_32k_1s",
    "dcase2025_task1_efficientat_32k_1s_repeat_to_10s",
    "dcase2025_task1_efficientat_32k_1s_zero_pad_to_10s",
    "dcase2025_task1_passt_32k_1s",
    "dcase2025_task1_passt_32k_1s_repeat_to_10s",
    "dcase2025_task1_passt_32k_1s_zero_pad_to_10s",
    "dymn_32k_10s_logmel128",
    "efficientat_32k_10s_logmel128",
    "efficientat_frontend",
    "panns_cnn14_16k_10s_logmel64",
    "panns_cnn14_32k_10s_logmel64",
    "panns_frontend",
    "passt_32k_10s_logmel128",
    "passt_frontend",
]
