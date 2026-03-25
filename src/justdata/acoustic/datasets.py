from justdata.core.registry import register_dataset


AUDIO_CLASSIFICATION_SINGLE_LABEL = "audio_classification_single_label"
AUDIO_CLASSIFICATION_MULTI_LABEL = "audio_classification_multi_label"
AUDIO_EVENT_DETECTION = "audio_event_detection"
AUDIO_RETRIEVAL = "audio_retrieval"
AUDIO_TEXT_PAIR = "audio_text_pair"
AUDIO_SELF_SUPERVISED = "audio_self_supervised"


register_dataset(
    "esc50",
    task_type=AUDIO_CLASSIFICATION_SINGLE_LABEL,
    modality="acoustic",
    pipeline_name="acoustic/classification",
    preset="audio_default_32k_logmel128",
)

register_dataset(
    "speech_commands",
    task_type=AUDIO_CLASSIFICATION_SINGLE_LABEL,
    modality="acoustic",
    pipeline_name="acoustic/classification",
    preset="audio_default_16k_waveform",
)

register_dataset(
    "audioset",
    task_type=AUDIO_CLASSIFICATION_MULTI_LABEL,
    modality="acoustic",
    pipeline_name="acoustic/classification",
    preset="audioset_32k_10s_logmel128",
)

register_dataset(
    "dcase2025_task1",
    task_type=AUDIO_CLASSIFICATION_SINGLE_LABEL,
    modality="acoustic",
    pipeline_name="acoustic/classification",
    preset="dcase2025_task1_native_44k_1s",
)


__all__ = [
    "AUDIO_CLASSIFICATION_MULTI_LABEL",
    "AUDIO_CLASSIFICATION_SINGLE_LABEL",
    "AUDIO_EVENT_DETECTION",
    "AUDIO_RETRIEVAL",
    "AUDIO_SELF_SUPERVISED",
    "AUDIO_TEXT_PAIR",
]
