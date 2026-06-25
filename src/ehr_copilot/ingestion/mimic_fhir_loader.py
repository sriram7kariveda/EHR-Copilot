"""MIMIC-IV Clinical Database on FHIR (NDJSON) loader.

Reads gzipped NDJSON files from the MIMIC-IV-on-FHIR distribution,
groups resources by patient, and produces :class:`ClinicalDocument`
objects ready for the chunking/indexing pipeline.

Expected data layout::

    data_dir/
        MimicPatient.ndjson.gz
        MimicEncounter.ndjson.gz
        MimicCondition.ndjson.gz
        MimicObservationLabevents.ndjson.gz
        MimicMedicationRequest.ndjson.gz
        MimicMedication.ndjson.gz
        MimicProcedure.ndjson.gz
        ...
"""

from __future__ import annotations

import gzip
import json
import logging
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from uuid import uuid4

from ehr_copilot.domain.clinical import (
    CodingEntry,
    Condition,
    Encounter,
    EncounterClass,
    MedicationRequest,
    Observation,
)
from ehr_copilot.domain.document import ClinicalDocument, DocumentType, NoteSection
from ehr_copilot.domain.patient import (
    Gender,
    PatientContext,
    PatientDemographics,
    PatientIdentifier,
)

logger = logging.getLogger(__name__)

# Resource files we load (in priority order for clinical Q&A).
_RESOURCE_FILES = {
    "patients": "MimicPatient.ndjson.gz",
    "encounters": "MimicEncounter.ndjson.gz",
    "encounters_ed": "MimicEncounterED.ndjson.gz",
    "encounters_icu": "MimicEncounterICU.ndjson.gz",
    "conditions": "MimicCondition.ndjson.gz",
    "conditions_ed": "MimicConditionED.ndjson.gz",
    "labs": "MimicObservationLabevents.ndjson.gz",
    "vitals_ed": "MimicObservationVitalSignsED.ndjson.gz",
    "med_requests": "MimicMedicationRequest.ndjson.gz",
    "medications": "MimicMedication.ndjson.gz",
    "procedures": "MimicProcedure.ndjson.gz",
}

_ENCOUNTER_CLASS_MAP: dict[str, EncounterClass] = {
    "AMB": EncounterClass.AMBULATORY,
    "EMER": EncounterClass.EMERGENCY,
    "IMP": EncounterClass.INPATIENT,
    "ACUTE": EncounterClass.INPATIENT,
    "SS": EncounterClass.OUTPATIENT,
    "OBSENC": EncounterClass.INPATIENT,
    "EW": EncounterClass.EMERGENCY,
}

_GENDER_MAP: dict[str, Gender] = {
    "male": Gender.MALE,
    "female": Gender.FEMALE,
    "other": Gender.OTHER,
    "unknown": Gender.UNKNOWN,
}


def _read_ndjson_gz(path: Path) -> list[dict]:
    """Read a gzipped NDJSON file and return a list of parsed dicts."""
    if not path.exists():
        logger.debug("NDJSON file not found, skipping: %s", path.name)
        return []
    records = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _patient_ref_id(resource: dict) -> str | None:
    """Extract the patient UUID from subject.reference."""
    ref = (resource.get("subject") or {}).get("reference", "")
    if ref.startswith("Patient/"):
        return ref[len("Patient/"):]
    return None


def _encounter_ref_id(resource: dict) -> str | None:
    """Extract the encounter UUID from encounter.reference."""
    ref = (resource.get("encounter") or {}).get("reference", "")
    if ref.startswith("Encounter/"):
        return ref[len("Encounter/"):]
    return None


def _parse_datetime(value: str | None) -> datetime | None:
    """Parse a FHIR datetime string, returning a timezone-naive datetime.

    MIMIC-FHIR uses timezone-aware strings (e.g. ``2180-05-06T22:23:00-04:00``)
    but our domain models and pipeline use naive datetimes, so we strip tzinfo.
    """
    if not value:
        return None
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(value, fmt)
            # Strip timezone to keep consistency with the rest of the pipeline.
            return dt.replace(tzinfo=None)
        except ValueError:
            continue
    return None


def _parse_date(value: str | None) -> date | None:
    """Parse a FHIR date string (YYYY-MM-DD)."""
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _first_coding(resource: dict, field: str = "code") -> CodingEntry | None:
    """Extract the first coding entry from a CodeableConcept field."""
    concept = resource.get(field)
    if not concept:
        return None
    codings = concept.get("coding", [])
    if not codings:
        return None
    c = codings[0]
    return CodingEntry(
        system=c.get("system", ""),
        code=c.get("code", ""),
        display=c.get("display", ""),
    )


class MimicFhirLoader:
    """Load patient data from MIMIC-IV Clinical Database on FHIR (NDJSON).

    Usage::

        loader = MimicFhirLoader(data_dir)
        patients = loader.list_patients()
        patient_ctx, documents, resources = loader.load_patient(patient_id)
    """

    def __init__(self, data_dir: Path | str) -> None:
        self._data_dir = Path(data_dir)
        if not self._data_dir.is_dir():
            raise FileNotFoundError(f"MIMIC-FHIR data directory not found: {self._data_dir}")

        # Lazily loaded indexes: patient_id -> list[resource_dict]
        self._patient_index: dict[str, dict] | None = None
        self._encounter_index: dict[str, list[dict]] | None = None
        self._condition_index: dict[str, list[dict]] | None = None
        self._lab_index: dict[str, list[dict]] | None = None
        self._vitals_index: dict[str, list[dict]] | None = None
        self._med_request_index: dict[str, list[dict]] | None = None
        self._procedure_index: dict[str, list[dict]] | None = None
        self._medication_lookup: dict[str, dict] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_patients(self) -> list[dict[str, str]]:
        """Return a list of ``{id, name, gender, birthDate}`` for all patients."""
        self._ensure_patients_loaded()
        assert self._patient_index is not None
        result = []
        for pid, rec in self._patient_index.items():
            names = rec.get("name", [])
            family = names[0].get("family", pid) if names else pid
            result.append({
                "id": pid,
                "name": family,
                "gender": rec.get("gender", "unknown"),
                "birthDate": rec.get("birthDate", ""),
            })
        return result

    def load_patient(
        self,
        patient_id: str,
    ) -> tuple[PatientContext, list[ClinicalDocument], dict]:
        """Load all data for a single patient and produce ClinicalDocuments.

        Parameters
        ----------
        patient_id:
            The FHIR Patient resource ID (UUID).

        Returns
        -------
        tuple
            ``(PatientContext, list[ClinicalDocument], resources_dict)``
        """
        self._ensure_all_loaded()
        assert self._patient_index is not None

        patient_rec = self._patient_index.get(patient_id)
        if patient_rec is None:
            raise ValueError(f"Patient not found: {patient_id}")

        # Build PatientContext
        patient_ctx = self._build_patient_context(patient_id, patient_rec)

        # Gather resources for this patient
        encounters = self._encounter_index.get(patient_id, [])  # type: ignore[union-attr]
        conditions = self._condition_index.get(patient_id, [])  # type: ignore[union-attr]
        labs = self._lab_index.get(patient_id, [])  # type: ignore[union-attr]
        vitals = self._vitals_index.get(patient_id, [])  # type: ignore[union-attr]
        med_requests = self._med_request_index.get(patient_id, [])  # type: ignore[union-attr]
        procedures = self._procedure_index.get(patient_id, [])  # type: ignore[union-attr]

        logger.info(
            "Patient %s: %d encounters, %d conditions, %d labs, %d vitals, "
            "%d med requests, %d procedures",
            patient_id, len(encounters), len(conditions), len(labs),
            len(vitals), len(med_requests), len(procedures),
        )

        # Update resource counts
        patient_ctx.resource_counts = {
            "encounters": len(encounters),
            "conditions": len(conditions),
            "labs": len(labs),
            "vitals": len(vitals),
            "medication_requests": len(med_requests),
            "procedures": len(procedures),
        }

        # Build encounter-indexed resource maps
        enc_conditions: dict[str, list[dict]] = defaultdict(list)
        for c in conditions:
            eid = _encounter_ref_id(c)
            if eid:
                enc_conditions[eid].append(c)

        enc_labs: dict[str, list[dict]] = defaultdict(list)
        for lab in labs:
            eid = _encounter_ref_id(lab)
            if eid:
                enc_labs[eid].append(lab)

        enc_vitals: dict[str, list[dict]] = defaultdict(list)
        for v in vitals:
            eid = _encounter_ref_id(v)
            if eid:
                enc_vitals[eid].append(v)

        enc_meds: dict[str, list[dict]] = defaultdict(list)
        for m in med_requests:
            eid = _encounter_ref_id(m)
            if eid:
                enc_meds[eid].append(m)

        enc_procs: dict[str, list[dict]] = defaultdict(list)
        for p in procedures:
            eid = _encounter_ref_id(p)
            if eid:
                enc_procs[eid].append(p)

        # Build ClinicalDocuments per encounter
        documents: list[ClinicalDocument] = []
        for enc_rec in encounters:
            enc_id = enc_rec["id"]
            doc = self._build_encounter_document(
                patient_id=patient_id,
                encounter=enc_rec,
                conditions=enc_conditions.get(enc_id, []),
                labs=enc_labs.get(enc_id, []),
                vitals=enc_vitals.get(enc_id, []),
                med_requests=enc_meds.get(enc_id, []),
                procedures=enc_procs.get(enc_id, []),
            )
            if doc.text.strip():
                documents.append(doc)

        # Build standalone lab summary if there are unlinked labs
        linked_enc_ids = {enc["id"] for enc in encounters}
        unlinked_labs = [
            lab for lab in labs
            if _encounter_ref_id(lab) not in linked_enc_ids
        ]
        if unlinked_labs:
            lab_doc = self._build_lab_summary_document(patient_id, unlinked_labs)
            if lab_doc.text.strip():
                documents.append(lab_doc)

        resources = {
            "encounters": encounters,
            "conditions": conditions,
            "labs": labs,
            "vitals": vitals,
            "med_requests": med_requests,
            "procedures": procedures,
        }

        logger.info(
            "Patient %s: produced %d clinical documents",
            patient_id, len(documents),
        )

        return patient_ctx, documents, resources

    # ------------------------------------------------------------------
    # Index loading
    # ------------------------------------------------------------------

    def _ensure_patients_loaded(self) -> None:
        if self._patient_index is not None:
            return
        path = self._data_dir / _RESOURCE_FILES["patients"]
        records = _read_ndjson_gz(path)
        self._patient_index = {rec["id"]: rec for rec in records}
        logger.info("Loaded %d patients from %s", len(self._patient_index), path.name)

    def _ensure_all_loaded(self) -> None:
        """Load and index all resource files."""
        self._ensure_patients_loaded()

        if self._encounter_index is not None:
            return  # Already loaded

        # Encounters (merge all encounter types)
        self._encounter_index = defaultdict(list)
        for key in ("encounters", "encounters_ed", "encounters_icu"):
            path = self._data_dir / _RESOURCE_FILES[key]
            for rec in _read_ndjson_gz(path):
                pid = _patient_ref_id(rec)
                if pid:
                    self._encounter_index[pid].append(rec)

        # Conditions
        self._condition_index = defaultdict(list)
        for key in ("conditions", "conditions_ed"):
            path = self._data_dir / _RESOURCE_FILES[key]
            for rec in _read_ndjson_gz(path):
                pid = _patient_ref_id(rec)
                if pid:
                    self._condition_index[pid].append(rec)

        # Labs
        self._lab_index = defaultdict(list)
        path = self._data_dir / _RESOURCE_FILES["labs"]
        for rec in _read_ndjson_gz(path):
            pid = _patient_ref_id(rec)
            if pid:
                self._lab_index[pid].append(rec)

        # Vitals
        self._vitals_index = defaultdict(list)
        path = self._data_dir / _RESOURCE_FILES["vitals_ed"]
        for rec in _read_ndjson_gz(path):
            pid = _patient_ref_id(rec)
            if pid:
                self._vitals_index[pid].append(rec)

        # Medication requests
        self._med_request_index = defaultdict(list)
        path = self._data_dir / _RESOURCE_FILES["med_requests"]
        for rec in _read_ndjson_gz(path):
            pid = _patient_ref_id(rec)
            if pid:
                self._med_request_index[pid].append(rec)

        # Medications (lookup by ID for resolving medicationReference)
        self._medication_lookup = {}
        path = self._data_dir / _RESOURCE_FILES["medications"]
        for rec in _read_ndjson_gz(path):
            self._medication_lookup[rec["id"]] = rec

        # Procedures
        self._procedure_index = defaultdict(list)
        path = self._data_dir / _RESOURCE_FILES["procedures"]
        for rec in _read_ndjson_gz(path):
            pid = _patient_ref_id(rec)
            if pid:
                self._procedure_index[pid].append(rec)

        total = sum(
            len(v) for idx in (
                self._encounter_index, self._condition_index,
                self._lab_index, self._vitals_index,
                self._med_request_index, self._procedure_index,
            ) for v in idx.values()
        )
        logger.info("MIMIC-FHIR index built: %d total resources across all patients", total)

    # ------------------------------------------------------------------
    # Document builders
    # ------------------------------------------------------------------

    def _build_patient_context(self, patient_id: str, rec: dict) -> PatientContext:
        """Build a PatientContext from a FHIR Patient resource."""
        names = rec.get("name", [])
        family = names[0].get("family", f"Patient_{patient_id[:8]}") if names else f"Patient_{patient_id[:8]}"
        given = names[0].get("given", []) if names else []

        gender_str = rec.get("gender", "unknown")
        gender = _GENDER_MAP.get(gender_str, Gender.UNKNOWN)

        birth_date = _parse_date(rec.get("birthDate"))

        deceased = rec.get("deceasedBoolean", False)
        deceased_date = _parse_date(rec.get("deceasedDateTime"))

        return PatientContext(
            patient_id=PatientIdentifier(
                system="urn:mimic-iv",
                value=patient_id,
                display=family,
            ),
            demographics=PatientDemographics(
                family_name=family,
                given_names=given,
                birth_date=birth_date,
                gender=gender,
                deceased=deceased or deceased_date is not None,
                deceased_date=deceased_date,
            ),
            session_id=str(uuid4()),
            source="mimic-iv-fhir",
        )

    def _build_encounter_document(
        self,
        patient_id: str,
        encounter: dict,
        conditions: list[dict],
        labs: list[dict],
        vitals: list[dict],
        med_requests: list[dict],
        procedures: list[dict],
    ) -> ClinicalDocument:
        """Build a ClinicalDocument for a single encounter."""
        enc_id = encounter["id"]
        period = encounter.get("period", {})
        period_start = _parse_datetime(period.get("start"))
        period_end = _parse_datetime(period.get("end"))

        # Encounter class
        enc_class_raw = encounter.get("class", {})
        enc_class_code = enc_class_raw.get("code", "OTHER") if isinstance(enc_class_raw, dict) else "OTHER"
        enc_class = _ENCOUNTER_CLASS_MAP.get(enc_class_code, EncounterClass.OTHER)

        # Encounter type
        enc_types = encounter.get("type", [])
        enc_type_display = ""
        if enc_types:
            codings = enc_types[0].get("coding", [])
            if codings:
                enc_type_display = codings[0].get("display", "")

        date_str = period_start.strftime("%Y-%m-%d %H:%M") if period_start else "Unknown date"
        title = f"{enc_class.value.title()} Encounter on {date_str}"

        sections: dict[NoteSection, str] = {}

        # Assessment/Plan: conditions
        if conditions:
            lines = []
            for c in conditions:
                coding = _first_coding(c)
                display = coding.display if coding else "Unknown condition"
                code_str = f" [{coding.code}]" if coding and coding.code else ""
                lines.append(f"- {display}{code_str}")
            sections[NoteSection.ASSESSMENT_PLAN] = (
                f"Diagnoses ({len(conditions)}):\n" + "\n".join(lines)
            )

        # Labs
        if labs:
            lines = []
            for lab in sorted(labs, key=lambda x: x.get("issued", "")):
                coding = _first_coding(lab)
                display = coding.display if coding else "Unknown test"
                vq = lab.get("valueQuantity", {})
                value = vq.get("value", "")
                unit = vq.get("unit") or vq.get("code", "")
                issued = lab.get("issued", "")[:10]
                if value:
                    lines.append(f"- {display}: {value} {unit} ({issued})")
                else:
                    vs = lab.get("valueString", "")
                    if vs:
                        lines.append(f"- {display}: {vs} ({issued})")
            if lines:
                sections[NoteSection.LABS_RESULTS] = (
                    f"Laboratory Results ({len(lines)}):\n" + "\n".join(lines)
                )

        # Vitals
        if vitals:
            lines = []
            for v in sorted(vitals, key=lambda x: x.get("effectiveDateTime", "")):
                coding = _first_coding(v)
                display = coding.display if coding else "Vital sign"
                # Handle component-based vitals (e.g., blood pressure)
                components = v.get("component", [])
                if components:
                    parts = []
                    for comp in components:
                        cc = _first_coding(comp)
                        vq = comp.get("valueQuantity", {})
                        val = vq.get("value", "")
                        unit = vq.get("unit") or vq.get("code", "")
                        label = cc.display if cc else "component"
                        if val:
                            parts.append(f"{label}: {val} {unit}")
                    if parts:
                        lines.append(f"- {display}: {', '.join(parts)}")
                else:
                    vq = v.get("valueQuantity", {})
                    val = vq.get("value", "")
                    unit = vq.get("unit") or vq.get("code", "")
                    if val:
                        lines.append(f"- {display}: {val} {unit}")
            if lines:
                sections[NoteSection.PHYSICAL_EXAM] = (
                    f"Vital Signs ({len(lines)}):\n" + "\n".join(lines)
                )

        # Medications
        if med_requests:
            lines = []
            for m in med_requests:
                med_name = self._resolve_medication_name(m)
                dosage = ""
                dosage_list = m.get("dosageInstruction", [])
                if dosage_list:
                    dosage = dosage_list[0].get("text", "")
                status = m.get("status", "")
                line = f"- {med_name}"
                if dosage:
                    line += f" ({dosage})"
                if status:
                    line += f" [{status}]"
                lines.append(line)
            sections[NoteSection.MEDICATIONS] = (
                f"Medications ({len(lines)}):\n" + "\n".join(lines)
            )

        # Procedures
        if procedures:
            lines = []
            for p in procedures:
                coding = _first_coding(p)
                display = coding.display if coding else "Unknown procedure"
                performed = (p.get("performedDateTime") or p.get("performedPeriod", {}).get("start", ""))[:10]
                lines.append(f"- {display} ({performed})" if performed else f"- {display}")
            sections[NoteSection.PROCEDURES] = (
                f"Procedures ({len(lines)}):\n" + "\n".join(lines)
            )

        # Build full text from sections
        text_parts = [title, ""]
        if enc_type_display:
            text_parts.append(f"Type: {enc_type_display}")
        if period_start:
            text_parts.append(f"Start: {date_str}")
        if period_end:
            text_parts.append(f"End: {period_end.strftime('%Y-%m-%d %H:%M')}")
        text_parts.append("")

        for section, content in sections.items():
            text_parts.append(content)
            text_parts.append("")

        return ClinicalDocument(
            document_id=f"mimic-enc-{enc_id}",
            patient_id=patient_id,
            document_type=DocumentType.ENCOUNTER_SUMMARY,
            title=title,
            text="\n".join(text_parts),
            encounter_id=enc_id,
            encounter_date=period_start,
            source_file="mimic-iv-fhir",
            sections=sections,
        )

    def _build_lab_summary_document(
        self,
        patient_id: str,
        labs: list[dict],
    ) -> ClinicalDocument:
        """Build a summary document for labs not linked to a specific encounter."""
        lines = []
        for lab in sorted(labs, key=lambda x: x.get("issued", "")):
            coding = _first_coding(lab)
            display = coding.display if coding else "Unknown test"
            vq = lab.get("valueQuantity", {})
            value = vq.get("value", "")
            unit = vq.get("unit") or vq.get("code", "")
            issued = lab.get("issued", "")[:10]
            if value:
                lines.append(f"- {display}: {value} {unit} ({issued})")
            else:
                vs = lab.get("valueString", "")
                if vs:
                    lines.append(f"- {display}: {vs} ({issued})")

        text = f"Standalone Laboratory Results ({len(lines)}):\n" + "\n".join(lines)

        return ClinicalDocument(
            document_id=f"mimic-labs-unlinked-{patient_id[:8]}",
            patient_id=patient_id,
            document_type=DocumentType.LAB_REPORT,
            title="Unlinked Laboratory Results",
            text=text,
            source_file="mimic-iv-fhir",
            sections={NoteSection.LABS_RESULTS: text},
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_medication_name(self, med_request: dict) -> str:
        """Resolve the medication display name from a MedicationRequest."""
        # Try inline medicationCodeableConcept
        concept = med_request.get("medicationCodeableConcept")
        if concept:
            codings = concept.get("coding", [])
            if codings and codings[0].get("display"):
                return codings[0]["display"]

        # Try medicationReference → lookup in Medication resources
        ref = (med_request.get("medicationReference") or {}).get("reference", "")
        if ref.startswith("Medication/") and self._medication_lookup:
            med_id = ref[len("Medication/"):]
            med_rec = self._medication_lookup.get(med_id)
            if med_rec:
                # Check identifiers for medication name
                for ident in med_rec.get("identifier", []):
                    if "medication-name" in ident.get("system", ""):
                        return ident.get("value", "Unknown medication")
                # Fallback to code display
                coding = _first_coding(med_rec)
                if coding and coding.display:
                    return coding.display

        return "Unknown medication"
