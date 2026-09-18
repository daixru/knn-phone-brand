"""Survey option sets shared by training and inference."""

# Full set encoded in the trained model (includes legacy CSV value).
MULTI_OPTIONS_FULL: tuple[str, ...] = (
    "Статус",
    "Камера",
    "Экосистема",
    "Надежность",
    "Мультивыбор",
)

# Shown in the UI — multi is implied by checkbox controls, not a separate answer.
MULTI_OPTIONS_UI: tuple[str, ...] = MULTI_OPTIONS_FULL[:-1]
