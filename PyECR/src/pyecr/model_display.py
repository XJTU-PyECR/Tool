"""Versioned presentation aliases. ModelID and serialized science stay canonical."""
from dataclasses import dataclass
from types import MappingProxyType

VERSION = "model-display/2.0.0"


@dataclass(frozen=True)
class ModelDisplay:
    number: str
    model_id: str
    title: str
    family: str

    @property
    def label(self):
        return f"{self.number} {self.title}"


MODEL_DISPLAY = (
    ModelDisplay("01", "M0_SINGLE_D", "M0_SINGLE_D", "single"),
    ModelDisplay("02", "M7_SINGLE_K", "M1_SINGLE_K", "single"),
    ModelDisplay("03", "M1_SINGLE_DK", "M2_SINGLE_DK", "single"),
    ModelDisplay("04", "M2_DUAL_HO_D", "M3_DUAL_H2O_D", "dual"),
    ModelDisplay("05", "M8_DUAL_K", "M4_DUAL_H2O_K", "dual"),
    ModelDisplay("06", "M3_DUAL_HO_DK", "M5_DUAL_H2O_DK", "dual"),
)
BY_ID = MappingProxyType({item.model_id: item for item in MODEL_DISPLAY})
_V1_TITLES = (
    "Single relaxation · fit D", "Single relaxation · analytical surface-controlled k", "Single relaxation · D+k",
    "Twofold relaxation · fit D", "Twofold relaxation · analytical surface-controlled k", "Twofold relaxation · D+k",
)
# Accept former callback text, but never show it in the new selector.
BY_LABEL = MappingProxyType({
    **{item.label: item.model_id for item in MODEL_DISPLAY},
    **{f"{item.number}  {title} [{item.model_id}]": item.model_id
       for item, title in zip(MODEL_DISPLAY, _V1_TITLES, strict=True)},
})


def model_id_for_display(value):
    value = str(getattr(value, "value", value))
    return value if value in BY_ID else BY_LABEL[value]


def display_label(value):
    return BY_ID[model_id_for_display(value)].label


def display_labels():
    return tuple(item.label for item in MODEL_DISPLAY)


def display_name(value):
    """Use aliases in prose/figures; leave auxiliary model names untouched."""
    value = str(getattr(value, "value", value))
    return BY_ID[value].title if value in BY_ID else value


def display_model_text(value):
    """Translate complete canonical tokens in a UI buffer in one pass."""
    import re
    return re.sub(r"\b(?:" + "|".join(map(re.escape, BY_ID)) + r")\b",
                  lambda match: display_name(match.group()), str(value))


def display_fit_worker_text(value):
    """Bare M0/M1/... stages mean legacy IDs only in original worker prose.

    Do not apply this to already formatted UI prose or serialized records.
    Full identifiers and bare stages are translated in the same pass, so the
    resulting menu aliases cannot be translated a second time.
    """
    import re
    aliases = {**{key: row.title for key, row in BY_ID.items()},
               **{key.split('_', 1)[0]: row.title for key, row in BY_ID.items()}}
    return re.sub(r'\b(?:'+'|'.join(map(re.escape, sorted(aliases, key=len, reverse=True)))+r')\b',
                  lambda m: aliases[m.group()], str(value))


def display_recommendation_text(value):
    """Format original recommendation prose with the actual selector labels.

    Bare scientific shorthand (M1, M2/M3) is understood only here, where the
    input is original scientific prose, never a JSON record or already-labelled
    UI text. A single substitution avoids alias-to-ID double translation.
    """
    import re
    aliases={**{key:row.label for key,row in BY_ID.items()},
             **{key.split('_',1)[0]:row.label for key,row in BY_ID.items()}}
    return re.sub(r'\b(?:'+'|'.join(map(re.escape,sorted(aliases,key=len,reverse=True)))+r')\b',
                  lambda match:aliases[match.group()],str(value))
