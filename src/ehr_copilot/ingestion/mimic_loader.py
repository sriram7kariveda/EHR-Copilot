"""MIMIC-III/IV data loader (stub).

This module defines the interface for loading patient records from MIMIC
CSV files.  The full implementation is planned for Week 6; for now the
class provides the public API surface and raises ``NotImplementedError``
for all data-loading methods.

Expected source directory layout (MIMIC-III)::

    source/
        ADMISSIONS.csv
        NOTEEVENTS.csv
        LABEVENTS.csv
        PATIENTS.csv
        ...
"""

from __future__ import annotations

import logging
from pathlib import Path

from ehr_copilot.domain.clinical import (
    CodingEntry,
    Condition,
    Encounter,
    EncounterClass,
    MedicationRequest,
    Observation,
)
from ehr_copilot.domain.document import ClinicalDocument, DocumentType
from ehr_copilot.domain.patient import (
    Gender,
    PatientContext,
    PatientDemographics,
    PatientIdentifier,
)
from ehr_copilot.ingestion.base import IngestorBase

logger = logging.getLogger(__name__)


class MIMICLoader(IngestorBase):
    """Load patient data from MIMIC-III / MIMIC-IV CSV exports.

    Parameters
    ----------
    version:
        Either ``"iii"`` or ``"iv"`` to select the MIMIC schema variant.
    subject_id:
        The MIMIC ``SUBJECT_ID`` to filter on.  If ``None``, the first
        patient found in the admissions table will be used.
    """

    def __init__(
        self,
        version: str = "iii",
        subject_id: int | None = None,
    ) -> None:
        if version not in ("iii", "iv"):
            raise ValueError(f"Unsupported MIMIC version: {version!r}")
        self.version = version
        self.subject_id = subject_id

    # ------------------------------------------------------------------
    # IngestorBase interface
    # ------------------------------------------------------------------

    def ingest(self, source: Path) -> list[ClinicalDocument]:
        """Load MIMIC data and return clinical documents.

        Parameters
        ----------
        source:
            Path to the directory containing MIMIC CSV files.
        """
        # TODO (Week 6): Implement full MIMIC ingestion pipeline
        self._validate_source(source)
        patient_ctx = self.load_patient(source)
        documents: list[ClinicalDocument] = []
        documents.extend(self.load_admissions(source, patient_ctx))
        documents.extend(self.load_notes(source, patient_ctx))
        documents.extend(self.load_labs(source, patient_ctx))
        return documents

    # ------------------------------------------------------------------
    # Public methods (to be implemented in Week 6)
    # ------------------------------------------------------------------

    def load_patient(self, source: Path) -> PatientContext:
        """Load patient demographics from PATIENTS.csv.

        Parameters
        ----------
        source:
            Directory containing MIMIC CSV files.

        Returns
        -------
        PatientContext
        """
        # TODO (Week 6): Parse PATIENTS.csv, build PatientContext
        raise NotImplementedError(
            "MIMICLoader.load_patient is not yet implemented (planned for Week 6)."
        )

    def load_admissions(
        self, source: Path, patient_ctx: PatientContext
    ) -> list[ClinicalDocument]:
        """Load admission records from ADMISSIONS.csv and produce encounter
        summary documents.

        Parameters
        ----------
        source:
            Directory containing MIMIC CSV files.
        patient_ctx:
            Patient context for the current session.

        Returns
        -------
        list[ClinicalDocument]
        """
        # TODO (Week 6): Parse ADMISSIONS.csv, create encounter-summary docs
        raise NotImplementedError(
            "MIMICLoader.load_admissions is not yet implemented (planned for Week 6)."
        )

    def load_notes(
        self, source: Path, patient_ctx: PatientContext
    ) -> list[ClinicalDocument]:
        """Load clinical notes from NOTEEVENTS.csv.

        Parameters
        ----------
        source:
            Directory containing MIMIC CSV files.
        patient_ctx:
            Patient context for the current session.

        Returns
        -------
        list[ClinicalDocument]
        """
        # TODO (Week 6): Parse NOTEEVENTS.csv, create ClinicalDocument per note
        raise NotImplementedError(
            "MIMICLoader.load_notes is not yet implemented (planned for Week 6)."
        )

    def load_labs(
        self, source: Path, patient_ctx: PatientContext
    ) -> list[ClinicalDocument]:
        """Load lab results from LABEVENTS.csv and produce structured documents.

        Parameters
        ----------
        source:
            Directory containing MIMIC CSV files.
        patient_ctx:
            Patient context for the current session.

        Returns
        -------
        list[ClinicalDocument]
        """
        # TODO (Week 6): Parse LABEVENTS.csv, create lab-report documents
        raise NotImplementedError(
            "MIMICLoader.load_labs is not yet implemented (planned for Week 6)."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_source(self, source: Path) -> None:
        """Check that required CSV files exist in *source*."""
        if not source.is_dir():
            raise FileNotFoundError(
                f"MIMIC source directory does not exist: {source}"
            )

        if self.version == "iii":
            required = ["ADMISSIONS.csv", "NOTEEVENTS.csv", "PATIENTS.csv"]
        else:
            required = ["admissions.csv", "discharge.csv", "patients.csv"]

        missing = [f for f in required if not (source / f).exists()]
        if missing:
            logger.warning(
                "Missing expected MIMIC files in %s: %s",
                source,
                ", ".join(missing),
            )
