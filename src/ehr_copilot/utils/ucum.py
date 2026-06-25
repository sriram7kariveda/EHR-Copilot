"""UCUM-style unit parsing and conversion using *pint*.

Provides a thin wrapper around :pypi:`pint` that adds awareness of common
clinical unit aliases that are not handled by pint out of the box.
"""

from __future__ import annotations

import pint


# ---------------------------------------------------------------------------
# Clinical unit alias mapping
# ---------------------------------------------------------------------------
# Maps informal / long-form clinical unit strings to the pint-compatible
# short form.  Keys are **lower-cased** for case-insensitive lookup.

_CLINICAL_ALIASES: dict[str, str] = {
    # --- Concentration ---
    "milligrams per deciliter": "mg/dL",
    "mg per dl": "mg/dL",
    "mg/dl": "mg/dL",
    "milligrams/deciliter": "mg/dL",
    "millimoles per liter": "mmol/L",
    "mmol per l": "mmol/L",
    "mmol/l": "mmol/L",
    "micrograms per deciliter": "ug/dL",
    "ug/dl": "ug/dL",
    "mcg/dl": "ug/dL",
    "nanograms per milliliter": "ng/mL",
    "ng/ml": "ng/mL",
    "milliequivalents per liter": "mEq/L",
    "meq/l": "mEq/L",
    "meq per l": "mEq/L",
    "grams per deciliter": "g/dL",
    "g/dl": "g/dL",
    "international units per liter": "IU/L",
    "iu/l": "IU/L",
    "units per liter": "U/L",
    "u/l": "U/L",

    # --- Mass ---
    "kilograms": "kg",
    "kilogram": "kg",
    "pounds": "lb",
    "pound": "lb",
    "lbs": "lb",
    "ounces": "oz",
    "ounce": "oz",
    "grams": "g",
    "gram": "g",
    "milligrams": "mg",
    "milligram": "mg",
    "micrograms": "ug",
    "microgram": "ug",
    "mcg": "ug",

    # --- Length / Height ---
    "centimeters": "cm",
    "centimeter": "cm",
    "meters": "m",
    "meter": "m",
    "inches": "inch",
    "in": "inch",
    "feet": "ft",
    "foot": "ft",

    # --- Volume ---
    "liters": "L",
    "liter": "L",
    "litres": "L",
    "litre": "L",
    "milliliters": "mL",
    "milliliter": "mL",
    "ml": "mL",
    "deciliters": "dL",
    "deciliter": "dL",
    "dl": "dL",

    # --- Percentage ---
    "percent": "%",
    "pct": "%",

    # --- Temperature ---
    "degrees celsius": "degC",
    "celsius": "degC",
    "degc": "degC",
    "degrees fahrenheit": "degF",
    "fahrenheit": "degF",
    "degf": "degF",

    # --- Pressure ---
    "millimeters of mercury": "mmHg",
    "mmhg": "mmHg",
    "mm hg": "mmHg",

    # --- Time ---
    "seconds": "s",
    "second": "s",
    "sec": "s",
    "minutes": "min",
    "minute": "min",
    "hours": "hr",
    "hour": "hr",
    "hr": "hr",
    "hrs": "hr",

    # --- Counts / rates ---
    "beats per minute": "count/min",
    "bpm": "count/min",
    "breaths per minute": "count/min",
}


class UnitConverter:
    """Clinical unit parser and converter backed by :pypi:`pint`.

    Instantiate once and reuse -- the underlying :class:`pint.UnitRegistry`
    is relatively expensive to create.

    Example
    -------
    >>> uc = UnitConverter()
    >>> uc.convert(100, "lb", "kg")  # doctest: +ELLIPSIS
    45.359...
    """

    def __init__(self) -> None:
        self._ureg = pint.UnitRegistry()
        # Pre-define percentage so that "%" is understood.
        try:
            self._ureg.define("percent = 1e-2 = %")
        except pint.errors.RedefinitionError:
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_unit(self, unit_str: str) -> str:
        """Normalize a clinical unit string to its canonical short form.

        The method first checks the built-in alias table (case-insensitive)
        and falls back to *pint*'s own parser.

        Parameters
        ----------
        unit_str:
            Raw unit string from a clinical note or lab result.

        Returns
        -------
        str
            The normalised unit string suitable for further conversion.
        """
        stripped = unit_str.strip()
        key = stripped.lower()

        # Check the alias table first.
        if key in _CLINICAL_ALIASES:
            return _CLINICAL_ALIASES[key]

        # Fall back to pint parsing -- return the compact string form.
        try:
            unit = self._ureg.parse_units(stripped)
            return f"{unit:~}"  # compact (abbreviated) form
        except pint.errors.UndefinedUnitError:
            # Return the original string if pint cannot parse it.
            return stripped

    def convert(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
    ) -> float | None:
        """Convert *value* from one unit to another.

        Parameters
        ----------
        value:
            Numeric value to convert.
        from_unit:
            Source unit (will be normalised via :meth:`parse_unit`).
        to_unit:
            Target unit (will be normalised via :meth:`parse_unit`).

        Returns
        -------
        float | None
            The converted numeric value, or ``None`` if the units are
            incompatible or unrecognised.
        """
        src = self.parse_unit(from_unit)
        dst = self.parse_unit(to_unit)
        try:
            quantity = self._ureg.Quantity(value, src)
            converted = quantity.to(dst)
            return float(converted.magnitude)
        except Exception:
            return None

    def are_compatible(self, unit1: str, unit2: str) -> bool:
        """Check whether two units are dimensionally compatible.

        Parameters
        ----------
        unit1:
            First unit string (will be normalised via :meth:`parse_unit`).
        unit2:
            Second unit string (will be normalised via :meth:`parse_unit`).

        Returns
        -------
        bool
            ``True`` if a conversion between the two units is possible.
        """
        u1 = self.parse_unit(unit1)
        u2 = self.parse_unit(unit2)
        try:
            q1 = self._ureg.Quantity(1, u1)
            q1.to(u2)
            return True
        except Exception:
            return False
