"""Synthea FHIR R4 Bundle parser.

Loads a FHIR R4 Bundle JSON file (as produced by Synthea) and converts it
into domain model objects: ``PatientContext``, ``ClinicalDocument``, and
the structured clinical resources (Encounter, Condition, Observation,
MedicationRequest).
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path

from fhir.resources.bundle import Bundle
from fhir.resources.condition import Condition as FHIRCondition
from fhir.resources.documentreference import DocumentReference as FHIRDocumentReference
from fhir.resources.encounter import Encounter as FHIREncounter
from fhir.resources.medicationrequest import MedicationRequest as FHIRMedicationRequest
from fhir.resources.observation import Observation as FHIRObservation
from fhir.resources.patient import Patient as FHIRPatient

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
from ehr_copilot.ingestion.base import IngestorBase

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------

_ENCOUNTER_CLASS_MAP: dict[str, EncounterClass] = {
    "AMB": EncounterClass.AMBULATORY,
    "ambulatory": EncounterClass.AMBULATORY,
    "EMER": EncounterClass.EMERGENCY,
    "emergency": EncounterClass.EMERGENCY,
    "IMP": EncounterClass.INPATIENT,
    "inpatient": EncounterClass.INPATIENT,
    "outpatient": EncounterClass.OUTPATIENT,
    "wellness": EncounterClass.WELLNESS,
}

_GENDER_MAP: dict[str, Gender] = {
    "male": Gender.MALE,
    "female": Gender.FEMALE,
    "other": Gender.OTHER,
    "unknown": Gender.UNKNOWN,
}


def _extract_coding(codeable_concept) -> CodingEntry | None:
    """Extract the first coding from a FHIR CodeableConcept."""
    if codeable_concept is None:
        return None
    codings = getattr(codeable_concept, "coding", None)
    if codings:
        c = codings[0]
        return CodingEntry(
            system=c.system or "",
            code=c.code or "",
            display=c.display or "",
        )
    text = getattr(codeable_concept, "text", None)
    if text:
        return CodingEntry(system="text", code="", display=text)
    return None


def _extract_codings_list(codeable_concepts) -> list[CodingEntry]:
    """Extract CodingEntry from a list of CodeableConcepts."""
    results: list[CodingEntry] = []
    if not codeable_concepts:
        return results
    for cc in codeable_concepts:
        entry = _extract_coding(cc)
        if entry is not None:
            results.append(entry)
    return results


def _ref_id(reference: str | None) -> str | None:
    """Extract the resource id from a FHIR reference.

    Handles formats like:
    - ``Encounter/abc-123``  → ``abc-123``
    - ``urn:uuid:abc-123``   → ``abc-123``
    - ``abc-123``            → ``abc-123``
    """
    if not reference:
        return None
    if reference.startswith("urn:uuid:"):
        return reference[len("urn:uuid:"):]
    if "/" in reference:
        return reference.split("/", 1)[1]
    return reference


def _parse_datetime(value) -> datetime | None:
    """Best-effort parse a FHIR dateTime value into a Python datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# FHIRBundleParser
# ---------------------------------------------------------------------------


class FHIRBundleParser(IngestorBase):
    """Parse a Synthea-generated FHIR R4 Bundle JSON file.

    The ``ingest`` method returns a flat list of ``ClinicalDocument`` objects.
    Use ``parse`` for a richer result that also includes the
    ``PatientContext`` and structured clinical resources.
    """

    def ingest(self, source: Path) -> list[ClinicalDocument]:
        """IngestorBase interface -- returns only documents."""
        _patient_ctx, documents, _resources = self.parse(source)
        return documents

    def parse(
        self, source: Path
    ) -> tuple[
        PatientContext,
        list[ClinicalDocument],
        dict[str, list],
    ]:
        """Parse a FHIR Bundle file and return rich results.

        Returns
        -------
        tuple
            (PatientContext, list[ClinicalDocument], resources_dict)
            where ``resources_dict`` maps resource type names to lists of
            domain model objects.
        """
        logger.info("Parsing FHIR bundle: %s", source)
        raw = json.loads(source.read_text(encoding="utf-8"))
        bundle = Bundle.model_validate(raw)

        # Categorise entries by resource type
        entries_by_type: dict[str, list] = {}
        for entry in bundle.entry or []:
            resource = entry.resource
            if resource is None:
                continue
            # fhir.resources v8+ uses get_resource_type(); v7 uses resource_type attribute
            if hasattr(resource, "get_resource_type"):
                rtype = resource.get_resource_type()
            else:
                rtype = resource.resource_type
            entries_by_type.setdefault(rtype, []).append(resource)

        # --- Patient ---
        fhir_patients = entries_by_type.get("Patient", [])
        if not fhir_patients:
            raise ValueError(f"No Patient resource found in {source}")
        patient_ctx = self._build_patient_context(fhir_patients[0], source)
        patient_id = patient_ctx.patient_id.value

        # --- Structured resources ---
        encounters = self._parse_encounters(
            entries_by_type.get("Encounter", []), patient_id
        )
        conditions = self._parse_conditions(
            entries_by_type.get("Condition", []), patient_id
        )
        observations = self._parse_observations(
            entries_by_type.get("Observation", []), patient_id
        )
        medications = self._parse_medications(
            entries_by_type.get("MedicationRequest", []), patient_id
        )

        resources: dict[str, list] = {
            "encounters": encounters,
            "conditions": conditions,
            "observations": observations,
            "medications": medications,
        }

        # Update resource counts on the patient context
        patient_ctx.resource_counts = {k: len(v) for k, v in resources.items()}

        # --- Documents ---
        documents: list[ClinicalDocument] = []

        # Encounter-summary documents (one per encounter)
        enc_lookup = {e.resource_id: e for e in encounters}
        cond_by_enc = self._group_by_encounter(conditions)
        obs_by_enc = self._group_by_encounter(observations)
        med_by_enc = self._group_by_encounter(medications)

        for enc in encounters:
            doc = self._build_encounter_summary(
                enc,
                cond_by_enc.get(enc.resource_id, []),
                obs_by_enc.get(enc.resource_id, []),
                med_by_enc.get(enc.resource_id, []),
                patient_id,
                str(source),
            )
            documents.append(doc)

        # Clinical notes from DocumentReference resources
        for doc_ref in entries_by_type.get("DocumentReference", []):
            note_doc = self._parse_document_reference(
                doc_ref, patient_id, enc_lookup, str(source)
            )
            if note_doc is not None:
                documents.append(note_doc)

        logger.info(
            "Parsed %d documents from %d encounters for patient %s",
            len(documents),
            len(encounters),
            patient_id,
        )

        return patient_ctx, documents, resources

    # ------------------------------------------------------------------
    # Patient
    # ------------------------------------------------------------------

    def _build_patient_context(
        self, fhir_patient: FHIRPatient, source: Path
    ) -> PatientContext:
        patient_id = fhir_patient.id or str(uuid.uuid4())

        # Demographics
        family_name = ""
        given_names: list[str] = []
        if fhir_patient.name:
            name = fhir_patient.name[0]
            family_name = name.family or ""
            given_names = list(name.given or [])

        birth_date = None
        if fhir_patient.birthDate:
            birth_date = fhir_patient.birthDate

        gender = _GENDER_MAP.get(
            (fhir_patient.gender or "").lower(), Gender.UNKNOWN
        )

        deceased = False
        deceased_date = None
        if fhir_patient.deceasedBoolean:
            deceased = True
        if fhir_patient.deceasedDateTime:
            deceased = True
            dt = _parse_datetime(fhir_patient.deceasedDateTime)
            if dt:
                deceased_date = dt.date()

        demographics = PatientDemographics(
            family_name=family_name,
            given_names=given_names,
            birth_date=birth_date,
            gender=gender,
            deceased=deceased,
            deceased_date=deceased_date,
        )

        return PatientContext(
            patient_id=PatientIdentifier(
                system="urn:synthea",
                value=patient_id,
                display=demographics.full_name,
            ),
            demographics=demographics,
            session_id=str(uuid.uuid4()),
            source="synthea",
        )

    # ------------------------------------------------------------------
    # Structured resources
    # ------------------------------------------------------------------

    def _parse_encounters(
        self, fhir_encounters: list[FHIREncounter], patient_id: str
    ) -> list[Encounter]:
        results: list[Encounter] = []
        for fe in fhir_encounters:
            enc_class_code = ""
            if fe.class_fhir:
                # R5: class_fhir is a list of CodeableConcepts
                if isinstance(fe.class_fhir, list) and fe.class_fhir:
                    first_cc = fe.class_fhir[0]
                    coding = _extract_coding(first_cc)
                    enc_class_code = coding.code if coding else ""
                elif hasattr(fe.class_fhir, "code"):
                    # R4 fallback: class_fhir is a single Coding
                    enc_class_code = fe.class_fhir.code or ""
            enc_class = _ENCOUNTER_CLASS_MAP.get(enc_class_code, EncounterClass.OTHER)

            type_code = None
            if fe.type:
                type_code = _extract_coding(fe.type[0])

            # R5 uses actualPeriod; R4 uses period
            period = getattr(fe, "actualPeriod", None) or getattr(fe, "period", None)
            period_start = _parse_datetime(period.start if period else None)
            period_end = _parse_datetime(period.end if period else None)

            # R5 uses reason; R4 uses reasonCode
            reason_codes_raw = getattr(fe, "reason", None) or getattr(fe, "reasonCode", None) or []
            if reason_codes_raw:
                # R5 reason is a list of CodeableReference; extract concept from each
                extracted: list[CodingEntry] = []
                for item in reason_codes_raw:
                    if hasattr(item, "concept") and item.concept:
                        entry = _extract_coding(item.concept)
                        if entry:
                            extracted.append(entry)
                    else:
                        entry = _extract_coding(item)
                        if entry:
                            extracted.append(entry)
                reason_codes = extracted
            else:
                reason_codes = []

            provider = None
            if fe.participant:
                for p in fe.participant:
                    if p.individual and p.individual.display:
                        provider = p.individual.display
                        break

            results.append(
                Encounter(
                    resource_id=fe.id or str(uuid.uuid4()),
                    patient_id=patient_id,
                    encounter_class=enc_class,
                    type_code=type_code,
                    period_start=period_start,
                    period_end=period_end,
                    reason_codes=reason_codes,
                    provider=provider,
                )
            )
        return results

    def _parse_conditions(
        self, fhir_conditions: list[FHIRCondition], patient_id: str
    ) -> list[Condition]:
        results: list[Condition] = []
        for fc in fhir_conditions:
            code = _extract_coding(fc.code)
            if code is None:
                continue

            onset = _parse_datetime(getattr(fc, "onsetDateTime", None))
            abatement = _parse_datetime(getattr(fc, "abatementDateTime", None))

            encounter_id = None
            if fc.encounter and fc.encounter.reference:
                encounter_id = _ref_id(fc.encounter.reference)

            clinical_status = "active"
            if fc.clinicalStatus:
                cs_coding = _extract_coding(fc.clinicalStatus)
                if cs_coding:
                    clinical_status = cs_coding.code or "active"

            from ehr_copilot.domain.clinical import ClinicalStatus

            status_map = {s.value: s for s in ClinicalStatus}
            status_enum = status_map.get(clinical_status, ClinicalStatus.ACTIVE)

            results.append(
                Condition(
                    resource_id=fc.id or str(uuid.uuid4()),
                    patient_id=patient_id,
                    code=code,
                    clinical_status=status_enum,
                    onset_date=onset,
                    abatement_date=abatement,
                    encounter_id=encounter_id,
                )
            )
        return results

    def _parse_observations(
        self, fhir_observations: list[FHIRObservation], patient_id: str
    ) -> list[Observation]:
        results: list[Observation] = []
        for fo in fhir_observations:
            code = _extract_coding(fo.code)
            if code is None:
                continue

            effective = _parse_datetime(getattr(fo, "effectiveDateTime", None))

            value_quantity = None
            value_unit = None
            value_string = None

            if fo.valueQuantity:
                value_quantity = fo.valueQuantity.value
                value_unit = fo.valueQuantity.unit or fo.valueQuantity.code
            elif fo.valueCodeableConcept:
                cc = _extract_coding(fo.valueCodeableConcept)
                value_string = cc.display if cc else None
            elif hasattr(fo, "valueString") and fo.valueString:
                value_string = fo.valueString

            ref_low = None
            ref_high = None
            if fo.referenceRange:
                rr = fo.referenceRange[0]
                if rr.low:
                    ref_low = rr.low.value
                if rr.high:
                    ref_high = rr.high.value

            encounter_id = None
            if fo.encounter and fo.encounter.reference:
                encounter_id = _ref_id(fo.encounter.reference)

            results.append(
                Observation(
                    resource_id=fo.id or str(uuid.uuid4()),
                    patient_id=patient_id,
                    code=code,
                    effective_date=effective,
                    value_quantity=value_quantity,
                    value_unit=value_unit,
                    value_string=value_string,
                    reference_range_low=ref_low,
                    reference_range_high=ref_high,
                    encounter_id=encounter_id,
                )
            )
        return results

    def _parse_medications(
        self, fhir_meds: list[FHIRMedicationRequest], patient_id: str
    ) -> list[MedicationRequest]:
        results: list[MedicationRequest] = []
        for fm in fhir_meds:
            # R5: medication is a CodeableReference with .concept
            # R4: medicationCodeableConcept is a CodeableConcept
            med_code = None
            if hasattr(fm, "medication") and fm.medication:
                # R5 CodeableReference
                if hasattr(fm.medication, "concept") and fm.medication.concept:
                    med_code = _extract_coding(fm.medication.concept)
                else:
                    med_code = _extract_coding(fm.medication)
            elif hasattr(fm, "medicationCodeableConcept") and fm.medicationCodeableConcept:
                med_code = _extract_coding(fm.medicationCodeableConcept)
            if med_code is None:
                continue

            authored = _parse_datetime(getattr(fm, "authoredOn", None))

            dosage_text = None
            if fm.dosageInstruction:
                di = fm.dosageInstruction[0]
                if hasattr(di, "text") and di.text:
                    dosage_text = di.text
                elif hasattr(di, "doseAndRate") and di.doseAndRate:
                    dr = di.doseAndRate[0]
                    if hasattr(dr, "doseQuantity") and dr.doseQuantity:
                        val = dr.doseQuantity.value
                        unit = dr.doseQuantity.unit or ""
                        dosage_text = f"{val} {unit}".strip()

            # R5: reason is list[CodeableReference]; R4: reasonCode is list[CodeableConcept]
            reason_codes_raw = getattr(fm, "reason", None) or getattr(fm, "reasonCode", None)
            reason_codes: list[CodingEntry] = []
            if reason_codes_raw:
                for item in reason_codes_raw:
                    if hasattr(item, "concept") and item.concept:
                        entry = _extract_coding(item.concept)
                        if entry:
                            reason_codes.append(entry)
                    else:
                        entry = _extract_coding(item)
                        if entry:
                            reason_codes.append(entry)

            encounter_id = None
            if fm.encounter and fm.encounter.reference:
                encounter_id = _ref_id(fm.encounter.reference)

            results.append(
                MedicationRequest(
                    resource_id=fm.id or str(uuid.uuid4()),
                    patient_id=patient_id,
                    medication_code=med_code,
                    status=fm.status or "active",
                    authored_on=authored,
                    dosage_text=dosage_text,
                    reason_codes=reason_codes,
                    encounter_id=encounter_id,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Document generation
    # ------------------------------------------------------------------

    @staticmethod
    def _group_by_encounter(items: list) -> dict[str, list]:
        grouped: dict[str, list] = {}
        for item in items:
            eid = getattr(item, "encounter_id", None)
            if eid:
                grouped.setdefault(eid, []).append(item)
        return grouped

    @staticmethod
    def _build_encounter_summary(
        encounter: Encounter,
        conditions: list[Condition],
        observations: list[Observation],
        medications: list[MedicationRequest],
        patient_id: str,
        source_file: str,
    ) -> ClinicalDocument:
        """Build a synthetic encounter-summary document from structured data."""
        sections: dict[NoteSection, str] = {}
        lines: list[str] = []

        # Header
        enc_type = encounter.type_code.display if encounter.type_code else "Encounter"
        enc_class = encounter.encounter_class.value.title()
        date_str = encounter.period_start.strftime("%Y-%m-%d") if encounter.period_start else "Unknown date"
        header = f"{enc_class} {enc_type} on {date_str}"
        if encounter.provider:
            header += f" (Provider: {encounter.provider})"
        lines.append(header)
        lines.append("")

        # Reasons for visit
        if encounter.reason_codes:
            reasons = ", ".join(r.display for r in encounter.reason_codes if r.display)
            if reasons:
                lines.append(f"Reason for visit: {reasons}")
                sections[NoteSection.CHIEF_COMPLAINT] = reasons
                lines.append("")

        # Conditions / diagnoses
        if conditions:
            cond_lines: list[str] = []
            for c in conditions:
                status = c.clinical_status.value
                onset = f" (onset: {c.onset_date.strftime('%Y-%m-%d')})" if c.onset_date else ""
                cond_lines.append(f"- {c.code.display} [{status}]{onset}")
            block = "\n".join(cond_lines)
            lines.append("Conditions:")
            lines.append(block)
            lines.append("")
            sections[NoteSection.ASSESSMENT_PLAN] = block

        # Observations / vitals / labs
        if observations:
            obs_lines: list[str] = []
            for o in observations:
                obs_lines.append(f"- {o.code.display}: {o.display_value}")
            block = "\n".join(obs_lines)
            lines.append("Observations:")
            lines.append(block)
            lines.append("")
            sections[NoteSection.LABS_RESULTS] = block

        # Medications
        if medications:
            med_lines: list[str] = []
            for m in medications:
                dose = f" ({m.dosage_text})" if m.dosage_text else ""
                med_lines.append(f"- {m.medication_code.display}{dose}")
            block = "\n".join(med_lines)
            lines.append("Medications:")
            lines.append(block)
            lines.append("")
            sections[NoteSection.MEDICATIONS] = block

        full_text = "\n".join(lines).strip()

        return ClinicalDocument(
            document_id=f"enc-summary-{encounter.resource_id}",
            patient_id=patient_id,
            document_type=DocumentType.ENCOUNTER_SUMMARY,
            title=header,
            text=full_text,
            encounter_id=encounter.resource_id,
            encounter_date=encounter.period_start,
            provider=encounter.provider,
            source_file=source_file,
            sections=sections,
        )

    @staticmethod
    def _parse_document_reference(
        doc_ref: FHIRDocumentReference,
        patient_id: str,
        encounters: dict[str, Encounter],
        source_file: str,
    ) -> ClinicalDocument | None:
        """Extract a clinical note from a FHIR DocumentReference."""
        # Get the text content from the attachment
        text = ""
        if doc_ref.content:
            for content_item in doc_ref.content:
                attachment = content_item.attachment
                if attachment and attachment.data:
                    import base64

                    try:
                        text = base64.b64decode(attachment.data).decode("utf-8")
                    except Exception:
                        logger.warning(
                            "Could not decode attachment for DocumentReference %s",
                            doc_ref.id,
                        )
                        continue

        if not text.strip():
            return None

        # Determine document type from category
        doc_type = DocumentType.CLINICAL_NOTE
        if doc_ref.type:
            type_coding = _extract_coding(doc_ref.type)
            if type_coding and type_coding.display:
                display_lower = type_coding.display.lower()
                if "discharge" in display_lower:
                    doc_type = DocumentType.DISCHARGE_SUMMARY
                elif "radiology" in display_lower or "imaging" in display_lower:
                    doc_type = DocumentType.RADIOLOGY_REPORT
                elif "pathology" in display_lower:
                    doc_type = DocumentType.PATHOLOGY_REPORT
                elif "lab" in display_lower:
                    doc_type = DocumentType.LAB_REPORT

        # Encounter link
        encounter_id = None
        encounter_date = None
        provider = None
        if doc_ref.context and hasattr(doc_ref.context, "encounter") and doc_ref.context.encounter:
            for enc_ref in doc_ref.context.encounter:
                eid = _ref_id(enc_ref.reference)
                if eid and eid in encounters:
                    encounter_id = eid
                    enc = encounters[eid]
                    encounter_date = enc.period_start
                    provider = enc.provider
                    break

        # Date
        if encounter_date is None and doc_ref.date:
            encounter_date = _parse_datetime(doc_ref.date)

        title_coding = _extract_coding(doc_ref.type)
        title = (title_coding.display if title_coding else None) or "Clinical Note"

        return ClinicalDocument(
            document_id=f"docref-{doc_ref.id or uuid.uuid4()}",
            patient_id=patient_id,
            document_type=doc_type,
            title=title,
            text=text,
            encounter_id=encounter_id,
            encounter_date=encounter_date,
            provider=provider,
            source_file=source_file,
        )
