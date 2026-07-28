"""
flags.py
========
Mapping Teamname -> ISO-3166-alpha-2, um in der UI Flaggen-Emojis zu zeigen.
Abgedeckt sind die Nationen, die in `matches_full.csv` regelmaessig vorkommen;
alles andere faellt sauber auf einen Ball zurueck.
"""

from __future__ import annotations

_ISO = {
    "Albania": "AL", "Algeria": "DZ", "Angola": "AO", "Argentina": "AR",
    "Armenia": "AM", "Australia": "AU", "Austria": "AT", "Azerbaijan": "AZ",
    "Bahrain": "BH", "Belarus": "BY", "Belgium": "BE", "Benin": "BJ",
    "Bolivia": "BO", "Bosnia and Herzegovina": "BA", "Brazil": "BR",
    "Bulgaria": "BG", "Burkina Faso": "BF", "Cameroon": "CM", "Canada": "CA",
    "Cape Verde": "CV", "Chile": "CL", "China PR": "CN", "Colombia": "CO",
    "Congo": "CG", "Costa Rica": "CR", "Croatia": "HR", "Cuba": "CU",
    "Curaçao": "CW", "Cyprus": "CY", "Czech Republic": "CZ", "Czechia": "CZ",
    "DR Congo": "CD", "Denmark": "DK", "Ecuador": "EC", "Egypt": "EG",
    "El Salvador": "SV", "England": "GB", "Equatorial Guinea": "GQ",
    "Estonia": "EE", "Ethiopia": "ET", "Finland": "FI", "France": "FR",
    "Gabon": "GA", "Gambia": "GM", "Georgia": "GE", "Germany": "DE",
    "Ghana": "GH", "Greece": "GR", "Guatemala": "GT", "Guinea": "GN",
    "Guinea-Bissau": "GW", "Haiti": "HT", "Honduras": "HN", "Hungary": "HU",
    "Iceland": "IS", "India": "IN", "Indonesia": "ID", "Iran": "IR",
    "Iraq": "IQ", "Israel": "IL", "Italy": "IT", "Ivory Coast": "CI",
    "Jamaica": "JM", "Japan": "JP", "Jordan": "JO", "Kazakhstan": "KZ",
    "Kenya": "KE", "Kosovo": "XK", "Kuwait": "KW", "Latvia": "LV",
    "Lebanon": "LB", "Libya": "LY", "Lithuania": "LT", "Luxembourg": "LU",
    "Malaysia": "MY", "Mali": "ML", "Malta": "MT", "Mauritania": "MR",
    "Mexico": "MX", "Moldova": "MD", "Montenegro": "ME", "Morocco": "MA",
    "Mozambique": "MZ", "Namibia": "NA", "Netherlands": "NL",
    "New Zealand": "NZ", "Nigeria": "NG", "North Macedonia": "MK",
    "Northern Ireland": "GB", "Norway": "NO", "Oman": "OM", "Panama": "PA",
    "Paraguay": "PY", "Peru": "PE", "Philippines": "PH", "Poland": "PL",
    "Portugal": "PT", "Qatar": "QA", "Republic of Ireland": "IE",
    "Romania": "RO", "Russia": "RU", "Saudi Arabia": "SA", "Scotland": "GB",
    "Senegal": "SN", "Serbia": "RS", "Sierra Leone": "SL", "Singapore": "SG",
    "Slovakia": "SK", "Slovenia": "SI", "South Africa": "ZA",
    "South Korea": "KR", "Spain": "ES", "Sudan": "SD", "Sweden": "SE",
    "Switzerland": "CH", "Syria": "SY", "Thailand": "TH", "Togo": "TG",
    "Trinidad and Tobago": "TT", "Tunisia": "TN", "Turkey": "TR",
    "Uganda": "UG", "Ukraine": "UA", "United Arab Emirates": "AE",
    "United States": "US", "Uruguay": "UY", "Uzbekistan": "UZ",
    "Venezuela": "VE", "Vietnam": "VN", "Wales": "GB", "Zambia": "ZM",
    "Zimbabwe": "ZW",
}

_OFFSET = 0x1F1E6 - ord("A")


def flag(team: str) -> str:
    """Flaggen-Emoji zum Teamnamen, sonst Ball."""
    code = _ISO.get(team)
    if not code or len(code) != 2:
        return "⚽"
    return "".join(chr(ord(ch) + _OFFSET) for ch in code)


def with_flag(team: str) -> str:
    return f"{flag(team)}  {team}"
