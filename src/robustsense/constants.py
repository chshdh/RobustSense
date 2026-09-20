"""Project-wide immutable labels and modality names."""

MODALITIES = (
    "phone_acc",
    "phone_gyro",
    "watch_acc",
    "location",
    "audio",
    "phone_state",
)

MODALITY_DIMS = {
    "phone_acc": 4,
    "phone_gyro": 4,
    "watch_acc": 5,
    "location": 3,
    "audio": 4,
    "phone_state": 3,
}

LABELS = (
    "SITTING",
    "LYING_DOWN",
    "OR_standing",
    "FIX_walking",
    "OR_exercise",
    "BICYCLING",
    "SLEEPING",
    "EATING",
    "TALKING",
    "COMPUTER_WORK",
    "LOC_home",
    "LOC_main_workplace",
    "OR_indoors",
    "OR_outside",
    "IN_A_CAR",
)
