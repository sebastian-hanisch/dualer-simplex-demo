"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Buttons (Standardmuster aus dem Demo-Portfolio)."""

import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import dsx_constants as C
import dsx_scenario as S


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


def _int_choice(options):
    def cast(value):
        value = int(value)
        if value not in options:
            raise ValueError(value)
        return value
    return cast


def _float_choice(options):
    def cast(value):
        value = float(value)
        if value not in options:
            raise ValueError(value)
        return value
    return cast


def _choice_from(options):
    def cast(value):
        value = str(value)
        if value not in options:
            raise ValueError(value)
        return value
    return cast


SETTING_SPECS = {
    "kind_select": SettingSpec("kind", _choice_from(S.KINDS), "centre"),
    "m_slider": SettingSpec("m", int, C.DEFAULT_M, C.M_MIN, C.M_MAX),
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "change_select": SettingSpec("change", _choice_from(C.CHANGES), "rhs"),
    "level_select": SettingSpec("level", int, C.DEFAULT_LEVEL["rhs"], 0, C.LEVEL_MAX),
    "resource_select": SettingSpec("res", int, 0, 0, C.INDEX_MAX),
    "var_select": SettingSpec("var", int, 0, 0, C.INDEX_MAX),
    "rule_select": SettingSpec("rule", _choice_from(C.RULES), C.DEFAULT_RULE),
    "dsx_step": SettingSpec("step", _int_choice(tuple(C.STEPS)), 2),
}
PRESET_KEYS = {"kind": "kind_select", "m": "m_slider", "n": "n_slider", "seed": "seed_input", "change": "change_select", "level": "level_select", "res": "resource_select", "var": "var_select",
               "rule": "rule_select", "step": "dsx_step", "pivot_k": "pivot_k"}
STEP_SLIDERS = ("pivot_k",)
STEPS = {}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


WIDGET_KEYS = {"m_slider": "m_widget", "n_slider": "n_widget", "seed_input": "seed_widget", "level_select": "level_widget"}


def store_from_widget(state_key):
    """Callback: übernimmt den Wert eines nur zeitweise sichtbaren Reglers in den dauerhaft gespeicherten Wert."""
    st.session_state[state_key] = st.session_state[WIDGET_KEYS[state_key]]


def push_to_widget(state_key):
    """Ist der Regler gerade sichtbar, muss ein geänderter gespeicherter Wert (Preset, Würfel) auch ihn selbst ändern."""
    widget_key = WIDGET_KEYS[state_key]
    if widget_key in st.session_state:
        st.session_state[widget_key] = st.session_state[state_key]


def apply_preset(name):
    """Setzt die Einstellungen; der Pivot-Regler (Schritt 1) wird geleert und nur gesetzt, wenn das Preset einen Wert ungleich 0 verlangt (dann steht es zugleich auf dem passenden Schritt, der Regler
    erscheint also im selben Lauf)."""
    for key in STEP_SLIDERS:
        st.session_state.pop(key, None)
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name] and not (state_key in STEP_SLIDERS and C.PRESETS[name][key] == 0):
            st.session_state[state_key] = C.PRESETS[name][key]
    for state_key in WIDGET_KEYS:
        push_to_widget(state_key)


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
    push_to_widget("seed_input")


def reset_level():
    """Callback der Änderungsart: die Größe springt auf den Standardwert der neuen Art."""
    st.session_state["level_select"] = C.DEFAULT_LEVEL[st.session_state["change_select"]]
    push_to_widget("level_select")
