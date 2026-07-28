"""
flags.py
========
Mapping Teamname -> Flaggen-Emoji, fuer die Team-Dropdowns im Dashboard.

Der Datensatz enthaelt 308 "Teams" - neben klassischen FIFA-Nationen auch
Regional- und Sondermannschaften (z.B. Catalonia, Isle of Wight, Republic of
St. Pauli) sowie einige Konfliktregionen (z.B. Kosovo, Artsakh). Drei Quellen
liefern die Flagge, in dieser Reihenfolge:

  1. OVERRIDES     - von Hand geprueft, fuer Faelle, in denen die
                     automatische Suche falsch oder politisch heikel waere.
  2. pycountry     - Fuzzy-Namenssuche gegen die ISO-3166-Laenderliste.
  3. Ball-Emoji    - Fallback fuer alles ohne eigene Landesflagge (Regionen,
                     Mikronationen, nicht-anerkannte Gebiete). Das ist kein
                     Fehler, sondern die einzig neutrale Darstellung.

BLOCKLIST verhindert, dass die Fuzzy-Suche einer Region unbeabsichtigt die
Flagge des uebergeordneten Staates gibt (z.B. "Kosovo" -> Serbien,
"Gagauzia" -> Moldau) - das waere sachlich falsch bzw. politisch fragwuerdig.
"""

from __future__ import annotations

from functools import lru_cache

import pycountry

_OFFSET = 0x1F1E6 - ord("A")


def _flag_from_alpha2(code: str) -> str:
    return "".join(chr(ord(ch) + _OFFSET) for ch in code.upper())


# --------------------------------------------------------------------------
# 1. Manuelle Overrides
# --------------------------------------------------------------------------
# a) Landesteile mit eigenem, in modernen Emoji-Sets unterstuetztem
#    Flaggen-Tag-Sequenz (funktioniert auf macOS/iOS zuverlaessig, auf
#    manchen Windows-/Linux-Schriftarten ggf. nicht).
_TAG_OFFSET = 0xE0000
_CANCEL_TAG = chr(0xE007F)


def _subdivision_flag(region_code: str) -> str:
    """z.B. 'gbeng' -> England-Flagge via Unicode-Tag-Sequenz."""
    base = "\U0001F3F4"  # schwarze Flagge als Traeger-Zeichen
    tags = "".join(chr(_TAG_OFFSET + ord(ch)) for ch in region_code)
    return base + tags + _CANCEL_TAG


_OVERRIDES: dict[str, str] = {
    # Britische Landesteile - eigene Nationalmannschaften, eigene Flaggen
    "England": _subdivision_flag("gbeng"),
    "Scotland": _subdivision_flag("gbsct"),
    "Wales": _subdivision_flag("gbwls"),
    # Nordirland hat keine unicodeseitig unterstuetzte eigene Flagge -
    # in der Praxis wird meist der Union Jack verwendet.
    "Northern Ireland": _flag_from_alpha2("GB"),

    # pycountry findet diese ueber Fuzzy-Suche nicht oder falsch
    "Curaçao": _flag_from_alpha2("CW"),
    "Sint Maarten": _flag_from_alpha2("SX"),
    "Ivory Coast": _flag_from_alpha2("CI"),
    "Cape Verde": _flag_from_alpha2("CV"),
    "DR Congo": _flag_from_alpha2("CD"),
    "Republic of Ireland": _flag_from_alpha2("IE"),
    "China PR": _flag_from_alpha2("CN"),
    # "Niger" ist als Teilstring in "Nigeria" enthalten - die Fuzzy-Suche
    # waehlt deshalb faelschlich Nigeria (NG) statt Niger (NE) als Top-Treffer.
    "Niger": _flag_from_alpha2("NE"),
}

# --------------------------------------------------------------------------
# 2. Blockliste
# --------------------------------------------------------------------------
# Regionen/Gebiete, bei denen die Fuzzy-Suche die Flagge eines (teils
# umstrittenen) uebergeordneten Staates liefern wuerde. Das waere hier
# irrefuehrend oder politisch heikel - deshalb bewusst kein Treffer.
_BLOCKLIST = {
    "Abkhazia", "Artsakh", "Darfur", "Gagauzia", "Kosovo", "Kurdistan",
    "Iraqi Kurdistan", "Matabeleland", "Zanzibar", "Crimea", "Donetsk PR",
    "Luhansk PR", "Northern Cyprus", "East Turkestan", "Somaliland",
    "Western Sahara", "Chechnya", "Chagos Islands",
    # Subnationale/regionale Ausstellungsteams - eigene Flagge existiert,
    # aber die Fuzzy-Suche wuerde die Flagge des Gesamtstaats liefern.
    "Galicia", "Gotland", "Orkney", "Shetland", "Quebec", "Provence",
    "Yorkshire", "Ynys Môn", "Isle of Wight", "Gozo", "Andalusia",
    "Basque Country", "Catalonia", "Corsica", "Brittany", "County of Nice",
    "Occitania", "Padania", "Franconia", "Raetia", "Canton Ticino",
    "Kabylia", "Biafra", "Ambazonia", "Barawa", "Cascadia", "Panjab",
    "Rhodes", "Menorca", "Canary Islands",
}


@lru_cache(maxsize=None)
def flag(team: str) -> str:
    """Flaggen-Emoji zum Teamnamen, sonst Ball als neutraler Platzhalter."""
    if team in _OVERRIDES:
        return _OVERRIDES[team]
    if team in _BLOCKLIST:
        return "⚽"
    try:
        match = pycountry.countries.search_fuzzy(team)[0]
        return _flag_from_alpha2(match.alpha_2)
    except LookupError:
        return "⚽"


def with_flag(team: str) -> str:
    return f"{flag(team)}  {team}"
