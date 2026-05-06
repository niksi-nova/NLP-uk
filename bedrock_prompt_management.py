"""
Prompt management and version tracking for Track B Bedrock summarization calls.

This module provides:
1. Local prompt template versioning with change log history.
2. Optional synchronization to Amazon Bedrock Prompt Management.
3. Deterministic A/B prompt version assignment.
4. Rollback support for prompt templates.
5. Prompt tracking metadata for reproducible LLM outputs.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from hipaa_compliance import create_secure_client

PROMPT_CONFIG_DIR = os.environ.get("PROMPT_MANAGEMENT_DIR", "prompt_management")
PROMPT_REGISTRY_PATH = os.path.join(PROMPT_CONFIG_DIR, "prompt_registry.json")
PROMPT_CHANGE_LOG_PATH = os.path.join(PROMPT_CONFIG_DIR, "prompt_change_log.json")
DEFAULT_MODEL_ID = os.environ.get(
    "BEDROCK_PROMPT_MODEL_ID", "us.anthropic.claude-sonnet-4-20250514-v1:0"
)


def _utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_weights(weights: Dict[str, Any]) -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    total = 0.0
    for key, value in weights.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric <= 0:
            continue
        normalized[key] = numeric
        total += numeric
    if total <= 0:
        return {}
    return {key: value / total for key, value in normalized.items()}


def _default_prompt_registry() -> Dict[str, Any]:
    return {
        "templates": {
            "medical_summarization": {
                "description": "Core medical summarization system prompt for Track B.",
                "active_version": "v3",
                "bedrock_prompt_id": None,
                "bedrock_prompt_arn": None,
                "versions": {
                    "v1": {
                        "created_at": "2026-04-09T00:00:00Z",
                        "created_by": "SYSTEM",
                        "rationale": "Baseline medically precise summarization template.",
                        "template": (
                            "You are a medical summarization assistant generating a {role_label} output.\n\n"
                            "{role_guidance}\n\n"
                            "<document_type>{document_type}</document_type>\n"
                            "{context_section}\n"
                            "<clinical_document>\n"
                            "{clinical_document}\n"
                            "</clinical_document>\n\n"
                            "Generate a structured JSON summary using exactly this schema:\n"
                            "{output_schema}\n"
                            "All dates in the summary field must use UK format DD/MM/YYYY "
                            "(e.g. 30/03/2026). Never write dates in Month-name or US format.\n"
                            "Medication names and doses must be written in full: drug name, "
                            "numeric dose with a space before the unit, and frequency as words "
                            "(e.g. 'Solifenacin 3 mg once daily', not '3mg OD').\n"
                        ),
                    },
                    "v2": {
                        "created_at": "2026-04-09T00:00:00Z",
                        "created_by": "SYSTEM",
                        "rationale": "Adds stronger ordering and contradiction checks for reproducibility.",
                        "template": (
                            "You are a senior clinical summarization assistant for {role_label} review.\n"
                            "Preserve chronology, avoid contradictions, and keep medication details explicit.\n\n"
                            "{role_guidance}\n\n"
                            "<document_type>{document_type}</document_type>\n"
                            "{context_section}\n"
                            "<clinical_document>\n"
                            "{clinical_document}\n"
                            "</clinical_document>\n\n"
                            "Produce strict JSON matching this schema (no markdown):\n"
                            "{output_schema}\n"
                            "All dates in the summary field must use UK format DD/MM/YYYY "
                            "(e.g. 30/03/2026). Never write dates in Month-name or US format.\n"
                            "Medication names and doses must be written in full: drug name, "
                            "numeric dose with a space before the unit, and frequency as words "
                            "(e.g. 'Solifenacin 3 mg once daily', not '3mg OD').\n"
                        ),
                    },
                    "v3": {
                        "created_at": "2026-05-06T00:00:00Z",
                        "created_by": "FINE_TUNE_ROUND_1",
                        "rationale": (
                            "Enforces GP-letter handover style for any NHS clinical document. "
                            "Constrains summary to 2-4 sentences opening with encounter type "
                            "and date. Preserves numeric precision. Enforces UK date format. "
                            "Ensures medication strings are written in full."
                        ),
                        "template": (
                            "You are a senior clinical summarization assistant producing a "
                            "{role_label} output for a UK NHS GP practice.\n\n"
                            "{role_guidance}\n\n"
                            "<document_type>{document_type}</document_type>\n"
                            "{context_section}\n"
                            "<clinical_document>\n"
                            "{clinical_document}\n"
                            "</clinical_document>\n\n"
                            "Rules for the summary field — follow these exactly for any "
                            "document type:\n\n"
                            "LENGTH AND STRUCTURE:\n"
                            "- Write exactly 2 to 4 sentences. Maximum 90 words total.\n"
                            "- Do not use bullet points or lists inside the summary field. "
                            "Write in continuous prose.\n\n"
                            "OPENING SENTENCE:\n"
                            "- The first sentence must identify the encounter type and the "
                            "encounter date in UK format DD/MM/YYYY. Derive both from the "
                            "document itself.\n"
                            "- Use the clinical event name as it appears in the document "
                            "(e.g. the department name, clinic name, or encounter description "
                            "followed by the date). Do not invent a label not present in "
                            "the source.\n\n"
                            "CONTENT TO INCLUDE:\n"
                            "- Key clinical findings from this encounter.\n"
                            "- Relevant negative findings that change the clinical picture "
                            "(e.g. no focal neurology, no infusion reactions).\n"
                            "- Investigations performed and their results if stated.\n"
                            "- The discharge plan, referral, or next step.\n\n"
                            "CONTENT TO EXCLUDE:\n"
                            "- Patient demographics (name, address, DOB, NHS number).\n"
                            "- GP or hospital contact details, salutations, signatures.\n"
                            "- Information not present in the source document.\n\n"
                            "PRECISION:\n"
                            "- Preserve all numeric values exactly as written in the source "
                            "(measurements, doses, ranges, scores, percentages).\n"
                            "- All dates must be in UK format DD/MM/YYYY.\n"
                            "- Medication names and doses must be written in full: drug name, "
                            "numeric dose with a space before the unit, frequency as full "
                            "words (e.g. 'Solifenacin 3 mg once daily').\n\n"
                            "Produce strict JSON matching this schema (no markdown, "
                            "no preamble, no trailing text):\n"
                            "{output_schema}\n"
                        ),
                    },
                },
            },
            "role_based_actions": {
                "description": "Prompt segment for role-specific follow-up action generation.",
                "active_version": "v3",
                "bedrock_prompt_id": None,
                "bedrock_prompt_arn": None,
                "versions": {
                    "v1": {
                        "created_at": "2026-04-09T00:00:00Z",
                        "created_by": "SYSTEM",
                        "rationale": "Baseline action extraction guidance.",
                        "template": (
                            "Action generation policy for {role_label}:\n"
                            "- Fill follow_up_actions with concise, practical actions.\n"
                            "- Include only actions explicitly supported by source text.\n"
                            "- Do not invent medications, dosages, diagnoses, or appointments.\n"
                        ),
                    },
                    "v2": {
                        "created_at": "2026-04-09T00:00:00Z",
                        "created_by": "SYSTEM",
                        "rationale": "Improves action prioritization and review readiness.",
                        "template": (
                            "Role-action policy for {role_label}:\n"
                            "- Prioritize urgent/safety-critical actions first.\n"
                            "- Keep each follow_up_actions item measurable and reviewable.\n"
                            "- If evidence is missing, return an empty action item list.\n"
                        ),
                    },
                    "v3": {
                        "created_at": "2026-05-06T00:00:00Z",
                        "created_by": "FINE_TUNE_ROUND_1",
                        "rationale": (
                            "Enforces verb-first actionable structure for any clinical "
                            "document type. Preserves timing information from source. "
                            "Differentiates medication changes as separate items. "
                            "Works for any NHS letter — does not rely on document-specific "
                            "knowledge."
                        ),
                        "template": (
                            "Role-action policy for {role_label}:\n\n"
                            "Generate follow_up_actions from the source document. Each "
                            "action item must follow ALL of these rules:\n\n"
                            "VERB RULE:\n"
                            "Every action must begin with an imperative verb. Choose the "
                            "most accurate verb from this list based on what the source "
                            "says: Refer, Arrange, Prescribe, Stop, Continue, Review, "
                            "Follow up, Discuss, Monitor, Advise, Repeat, Request, "
                            "Check, Notify, Counsel, Ensure.\n\n"
                            "CONTENT RULE:\n"
                            "Each action must answer what needs to happen, and where or "
                            "to whom if stated in the source. Write as a single complete "
                            "sentence. Do not use vague nouns like 'referral' or "
                            "'review' as the entire action — always state what the "
                            "referral or review is for.\n\n"
                            "TIMING RULE:\n"
                            "If the source states any time interval or deadline for an "
                            "action (e.g. 'in 6 months', 'in 4 weeks', 'before next "
                            "appointment', 'annually'), include that interval verbatim "
                            "in the action_text. The structured due_date field for that "
                            "action must be computed by adding that interval to the "
                            "letter date. If no timing is stated, leave due_date as "
                            "the default.\n\n"
                            "MEDICATION CHANGE RULE:\n"
                            "If the source contains any medication start, stop, dose "
                            "change, or switch, emit it as a SEPARATE action item. "
                            "Prefix the action_text with exactly 'Medication change: ' "
                            "followed by the instruction as written in the source "
                            "(drug name, dose with space before unit, frequency in full "
                            "words, route if stated).\n\n"
                            "SAFETY ADVICE RULE:\n"
                            "If the source contains driving restrictions, alcohol advice, "
                            "smoking advice, or falls precautions directed at the GP or "
                            "patient, emit each as a separate action item prefixed with "
                            "'Advise: '.\n\n"
                            "EVIDENCE RULE:\n"
                            "Only include actions explicitly supported by the source "
                            "document. If no follow-up actions are stated or implied, "
                            "return an empty array. Do not invent actions.\n\n"
                            "LIMIT: Maximum 8 action items. Order by clinical urgency, "
                            "most urgent first.\n"
                        ),
                    },
                },
            },
            "error_correction": {
                "description": "Prompt segment to minimize hallucination and enforce correction policy.",
                "active_version": "v1",
                "bedrock_prompt_id": None,
                "bedrock_prompt_arn": None,
                "versions": {
                    "v1": {
                        "created_at": "2026-04-09T00:00:00Z",
                        "created_by": "SYSTEM",
                        "rationale": "Baseline anti-hallucination controls.",
                        "template": (
                            "Strict reliability rules:\n"
                            "- Use only facts explicitly present in the document/context.\n"
                            '- If data is missing, output "Not specified" or empty arrays.\n'
                            "- Do not infer unsupported clinical facts.\n"
                            "- Return valid JSON only.\n"
                        ),
                    },
                    "v2": {
                        "created_at": "2026-04-09T00:00:00Z",
                        "created_by": "SYSTEM",
                        "rationale": "Adds explicit contradiction guardrails and confidence discipline.",
                        "template": (
                            "Reliability and correction controls:\n"
                            "- Never contradict source text.\n"
                            "- Reject unsupported assumptions; keep unknowns as Not specified.\n"
                            "- Calibrate confidence_score conservatively when source data is incomplete.\n"
                            "- Return valid JSON only.\n"
                        ),
                    },
                },
            },
            "summary_self_critique": {
                "description": (
                    "One-pass self-critique for Track B medium-confidence outputs. "
                    "Triggered only when unified confidence is in the medium band "
                    "(0.60 <= score < threshold). "
                    "This template adds one Bedrock API call per document-role pair "
                    "when triggered. Monitor usage in the CloudWatch cost dashboard."
                ),
                "active_version": "v1",
                "bedrock_prompt_id": None,
                "bedrock_prompt_arn": None,
                "versions": {
                    "v1": {
                        "created_at": "2026-05-06T00:00:00Z",
                        "created_by": "FINE_TUNE_ROUND_1",
                        "rationale": (
                            "Single self-critique pass to recover medium-confidence "
                            "outputs before routing to human review. General-purpose: "
                            "works for any NHS clinical document type. Expected to "
                            "resolve 30-50 percent of medium-band cases automatically."
                        ),
                        "template": (
                            "You previously produced this clinical summary for a "
                            "{role_label} audience:\n\n"
                            "<previous_summary>\n"
                            "{previous_summary}\n"
                            "</previous_summary>\n\n"
                            "The source document is:\n"
                            "<clinical_document>\n"
                            "{clinical_document}\n"
                            "</clinical_document>\n\n"
                            "Perform a strict self-critique. Check for ALL of the "
                            "following error types. Fix every error you find:\n\n"
                            "1. UNSUPPORTED CONTENT: Any sentence or claim in the "
                            "summary that is not directly supported by the source "
                            "document. Remove it.\n\n"
                            "2. NUMERIC ERRORS: Any numeric value, measurement, dose, "
                            "date, or range that differs from or is absent from the "
                            "source. Correct it to match the source exactly.\n\n"
                            "3. LATERALITY ERRORS: Any side (left/right/bilateral) "
                            "or anatomical sub-site stated in the source but missing "
                            "or wrong in the summary. Add or correct it.\n\n"
                            "4. MISSING ACTIONS: Any follow-up action, referral, "
                            "investigation, or medication change the source explicitly "
                            "states that is absent from follow_up_actions. Add it.\n\n"
                            "5. MEDICATION FORMAT ERRORS: Any medication name or dose "
                            "not written as: drug name, numeric dose, space, unit, "
                            "frequency in full words. Correct the format.\n\n"
                            "6. DATE FORMAT ERRORS: Any date not in UK format "
                            "DD/MM/YYYY. Correct it.\n\n"
                            "7. OPENING SENTENCE: If the first sentence of summary "
                            "does not state the encounter type and encounter date, "
                            "rewrite it so it does.\n\n"
                            "If you find no errors, return the previous summary "
                            "unchanged.\n\n"
                            "Return valid JSON conforming to exactly the same schema "
                            "as the previous summary. No markdown, no preamble, "
                            "no explanation — only the JSON output.\n"
                        ),
                    },
                },
            },
        },

        "ab_tests": {
            "medical_summarization": {
                "enabled": False,
                "weights": {"v1": 0.0, "v2": 0.0, "v3": 1.0},
                "salt": "trackb-medical-summary-exp",
            },
            "role_based_actions": {
                "enabled": False,
                "weights": {"v1": 0.0, "v2": 0.0, "v3": 1.0},
                "salt": "trackb-role-actions-exp",
            },
            "error_correction": {
                "enabled": False,
                "weights": {"v1": 0.5, "v2": 0.5},
                "salt": "trackb-error-correction-exp",
            },
            "summary_self_critique": {
                "enabled": False,
                "weights": {"v1": 1.0},
                "salt": "trackb-self-critique-exp",
            },
        },
        "last_updated": _utc_now(),
    }


class BedrockPromptManager:
    """Prompt manager with local version control and optional Bedrock sync."""

    def __init__(
        self,
        registry_path: str = PROMPT_REGISTRY_PATH,
        change_log_path: str = PROMPT_CHANGE_LOG_PATH,
        model_id: str = DEFAULT_MODEL_ID,
        region_name: str = "us-east-1",
        sync_enabled: Optional[bool] = None,
        auto_snapshot: Optional[bool] = None,
    ):
        self.registry_path = registry_path
        self.change_log_path = change_log_path
        self.model_id = model_id
        self.region_name = region_name
        self.sync_enabled = _normalize_bool(
            sync_enabled, default=_normalize_bool(os.getenv("PROMPT_MGMT_SYNC_ENABLED"), False)
        )
        self.auto_snapshot = _normalize_bool(
            auto_snapshot,
            default=_normalize_bool(os.getenv("PROMPT_MGMT_AUTO_SNAPSHOT"), False),
        )
        self._sync_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._prompt_client = None
        self._ensure_registry_files()
        self.registry = self._load_json(self.registry_path, default=_default_prompt_registry())
        self._ensure_registry_defaults()

    @staticmethod
    def _load_json(path: str, default: Any) -> Any:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return default

    @staticmethod
    def _write_json(path: str, payload: Any) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _ensure_registry_files(self) -> None:
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        if not os.path.exists(self.registry_path):
            self._write_json(self.registry_path, _default_prompt_registry())
        if not os.path.exists(self.change_log_path):
            self._write_json(self.change_log_path, [])

    def _ensure_registry_defaults(self) -> None:
        defaults = _default_prompt_registry()
        merged = self.registry or {}
        merged.setdefault("templates", {})
        merged.setdefault("ab_tests", {})
        for name, template_cfg in defaults["templates"].items():
            merged["templates"].setdefault(name, template_cfg)
        for name, ab_cfg in defaults["ab_tests"].items():
            merged["ab_tests"].setdefault(name, ab_cfg)
        merged["last_updated"] = _utc_now()
        self.registry = merged
        self._persist_registry()

    def _persist_registry(self) -> None:
        self.registry["last_updated"] = _utc_now()
        self._write_json(self.registry_path, self.registry)

    def _append_change_log(self, entry: Dict[str, Any]) -> None:
        history = self._load_json(self.change_log_path, default=[])
        if not isinstance(history, list):
            history = []
        history.append(entry)
        self._write_json(self.change_log_path, history)

    def _template_config(self, template_name: str) -> Dict[str, Any]:
        templates = self.registry.get("templates", {})
        if template_name not in templates:
            raise KeyError(f"Unknown prompt template: {template_name}")
        return templates[template_name]

    def list_versions(self, template_name: str) -> List[str]:
        cfg = self._template_config(template_name)
        versions = cfg.get("versions", {})
        return sorted(list(versions.keys()))

    def set_active_version(
        self,
        template_name: str,
        version: str,
        rationale: str,
        changed_by: str = "SYSTEM",
    ) -> None:
        cfg = self._template_config(template_name)
        versions = cfg.get("versions", {})
        if version not in versions:
            raise ValueError(f"Version '{version}' not found for template '{template_name}'")

        previous_version = cfg.get("active_version")
        cfg["active_version"] = version
        self._persist_registry()
        self._append_change_log(
            {
                "timestamp": _utc_now(),
                "change_type": "SET_ACTIVE_VERSION",
                "template": template_name,
                "before_version": previous_version,
                "after_version": version,
                "changed_by": changed_by,
                "rationale": rationale,
            }
        )

    def rollback_to_version(
        self,
        template_name: str,
        version: str,
        rationale: str,
        changed_by: str = "SYSTEM",
    ) -> None:
        cfg = self._template_config(template_name)
        previous_version = cfg.get("active_version")
        self.set_active_version(
            template_name=template_name,
            version=version,
            rationale=rationale,
            changed_by=changed_by,
        )
        self._append_change_log(
            {
                "timestamp": _utc_now(),
                "change_type": "ROLLBACK",
                "template": template_name,
                "before_version": previous_version,
                "after_version": version,
                "changed_by": changed_by,
                "rationale": rationale,
            }
        )

    def configure_ab_test(
        self,
        template_name: str,
        enabled: bool,
        weights: Optional[Dict[str, Any]] = None,
        salt: Optional[str] = None,
        changed_by: str = "SYSTEM",
        rationale: str = "",
    ) -> Dict[str, Any]:
        cfg = self._template_config(template_name)
        versions = set(cfg.get("versions", {}).keys())

        existing = self.registry["ab_tests"].get(
            template_name, {"enabled": False, "weights": {}, "salt": f"{template_name}-exp"}
        )
        next_cfg = dict(existing)
        next_cfg["enabled"] = bool(enabled)

        if weights is not None:
            normalized = _normalize_weights(weights)
            missing = set(normalized.keys()) - versions
            if missing:
                raise ValueError(
                    f"A/B weights reference unknown versions for {template_name}: {sorted(list(missing))}"
                )
            next_cfg["weights"] = normalized
        if salt:
            next_cfg["salt"] = salt

        self.registry["ab_tests"][template_name] = next_cfg
        self._persist_registry()
        self._append_change_log(
            {
                "timestamp": _utc_now(),
                "change_type": "CONFIGURE_AB_TEST",
                "template": template_name,
                "changed_by": changed_by,
                "rationale": rationale,
                "ab_test": next_cfg,
            }
        )
        return next_cfg

    def _resolve_version(
        self,
        template_name: str,
        document_id: str,
        role_key: str,
        forced_version: Optional[str] = None,
    ) -> Tuple[str, str]:
        cfg = self._template_config(template_name)
        versions = cfg.get("versions", {})

        if forced_version:
            if forced_version not in versions:
                raise ValueError(
                    f"Forced version '{forced_version}' not found for template '{template_name}'"
                )
            return forced_version, "forced"

        ab_cfg = self.registry.get("ab_tests", {}).get(template_name, {})
        if ab_cfg.get("enabled"):
            weights = _normalize_weights(ab_cfg.get("weights", {}))
            if weights:
                salt = ab_cfg.get("salt", template_name)
                fingerprint = (
                    f"{template_name}|{document_id}|{role_key}|{salt}".encode("utf-8")
                )
                random_value = int(hashlib.sha256(fingerprint).hexdigest(), 16) / float(
                    16**64 - 1
                )
                cumulative = 0.0
                for version, weight in sorted(weights.items()):
                    cumulative += weight
                    if random_value <= cumulative:
                        return version, "ab_test"

        return cfg.get("active_version", "v1"), "active"

    def render_template(
        self,
        template_name: str,
        variables: Dict[str, Any],
        document_id: str,
        role_key: str,
        forced_version: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        cfg = self._template_config(template_name)
        version, selection_mode = self._resolve_version(
            template_name=template_name,
            document_id=document_id,
            role_key=role_key,
            forced_version=forced_version,
        )
        template_record = cfg["versions"][version]
        rendered = template_record["template"].format(**variables)

        sync_result = self.sync_template_to_bedrock(template_name, version)
        tracking = {
            "template": template_name,
            "version": version,
            "selection_mode": selection_mode,
            "template_hash": _stable_hash(template_record["template"]),
            "rendered_prompt_hash": _stable_hash(rendered),
            "bedrock_prompt_id": cfg.get("bedrock_prompt_id"),
            "bedrock_prompt_arn": cfg.get("bedrock_prompt_arn"),
            "bedrock_sync_status": sync_result.get("status", "skipped"),
            "version_rationale": template_record.get("rationale", ""),
        }
        if sync_result.get("error"):
            tracking["bedrock_sync_error"] = sync_result["error"]
        if sync_result.get("bedrock_version"):
            tracking["bedrock_prompt_version"] = sync_result["bedrock_version"]
        return rendered, tracking

    def compose_track_b_prompt(
        self,
        document_id: str,
        role_key: str,
        role_guidance: str,
        document_type: str,
        clinical_document: str,
        retrieved_context: List[str],
        output_schema: Dict[str, Any],
        forced_versions: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        forced_versions = forced_versions or {}
        context_section = ""
        if retrieved_context:
            context_lines = [f"{idx + 1}. {ctx}" for idx, ctx in enumerate(retrieved_context[:3])]
            context_section = "<relevant_medical_context>\n" + "\n".join(context_lines) + "\n</relevant_medical_context>"

        vars_common = {
            "role_label": role_key,
            "role_guidance": role_guidance,
            "document_type": document_type,
            "context_section": context_section,
            "clinical_document": clinical_document[:12000],
            "output_schema": json.dumps(output_schema, indent=2),
        }

        sequence = ["medical_summarization", "role_based_actions", "error_correction"]
        parts: List[str] = []
        component_tracking: Dict[str, Dict[str, Any]] = {}
        for template_name in sequence:
            rendered, tracking = self.render_template(
                template_name=template_name,
                variables=vars_common,
                document_id=document_id,
                role_key=role_key,
                forced_version=forced_versions.get(template_name),
            )
            parts.append(rendered)
            component_tracking[template_name] = tracking

        final_prompt = "\n\n".join(parts).strip()
        tracking_summary = {
            "document_id": document_id,
            "role": role_key,
            "generated_at": _utc_now(),
            "components": component_tracking,
            "selected_versions": {
                name: details["version"] for name, details in component_tracking.items()
            },
            "selection_modes": {
                name: details["selection_mode"] for name, details in component_tracking.items()
            },
            "final_prompt_hash": _stable_hash(final_prompt),
            "truncated": len(clinical_document) > 12000,
        }
        return final_prompt, tracking_summary

    def _get_prompt_client(self):
        if not self.sync_enabled:
            return None
        if self._prompt_client is not None:
            return self._prompt_client
        try:
            self._prompt_client = create_secure_client(
                "bedrock-agent", region_name=self.region_name
            )
        except Exception:
            self._prompt_client = None
        return self._prompt_client

    def _find_prompt_summary(self, prompt_name: str) -> Optional[Dict[str, Any]]:
        client = self._get_prompt_client()
        if client is None:
            return None

        token = None
        while True:
            kwargs: Dict[str, Any] = {"maxResults": 100}
            if token:
                kwargs["nextToken"] = token
            response = client.list_prompts(**kwargs)
            for summary in response.get("promptSummaries", []):
                if summary.get("name") == prompt_name and summary.get("version") == "DRAFT":
                    return summary
            token = response.get("nextToken")
            if not token:
                break
        return None

    def sync_template_to_bedrock(self, template_name: str, version: str) -> Dict[str, Any]:
        cache_key = (template_name, version)
        if cache_key in self._sync_cache:
            return self._sync_cache[cache_key]

        if not self.sync_enabled:
            result = {"status": "skipped"}
            self._sync_cache[cache_key] = result
            return result

        client = self._get_prompt_client()
        if client is None:
            result = {"status": "skipped", "error": "bedrock-agent client unavailable"}
            self._sync_cache[cache_key] = result
            return result

        cfg = self._template_config(template_name)
        version_record = cfg.get("versions", {}).get(version)
        if not version_record:
            result = {"status": "error", "error": f"version '{version}' not found"}
            self._sync_cache[cache_key] = result
            return result

        prompt_name = f"nlp-uk-{template_name}"
        variant_payload = {
            "name": version,
            "templateType": "TEXT",
            "templateConfiguration": {"text": {"text": version_record["template"]}},
            "modelId": self.model_id,
            "inferenceConfiguration": {
                "text": {"temperature": 0.1, "maxTokens": 2000}
            },
            "metadata": [{"key": "version", "value": version}],
        }

        try:
            existing = self._find_prompt_summary(prompt_name)
            if existing and existing.get("id"):
                prompt_id = existing["id"]
                update_response = client.update_prompt(
                    promptIdentifier=prompt_id,
                    name=prompt_name,
                    description=cfg.get("description", ""),
                    defaultVariant=version,
                    variants=[variant_payload],
                )
                prompt_arn = update_response.get("arn", existing.get("arn"))
            else:
                create_response = client.create_prompt(
                    name=prompt_name,
                    description=cfg.get("description", ""),
                    defaultVariant=version,
                    variants=[variant_payload],
                    tags={"project": "NLP-uk", "component": "track-b"},
                )
                prompt_id = create_response.get("id")
                prompt_arn = create_response.get("arn")

            bedrock_prompt_version = None
            if self.auto_snapshot and prompt_id:
                snapshot = client.create_prompt_version(
                    promptIdentifier=prompt_id,
                    description=f"Snapshot for {template_name}:{version}",
                    tags={"project": "NLP-uk", "template": template_name},
                )
                bedrock_prompt_version = snapshot.get("version")

            cfg["bedrock_prompt_id"] = prompt_id
            cfg["bedrock_prompt_arn"] = prompt_arn
            self._persist_registry()
            result = {
                "status": "synced",
                "bedrock_prompt_id": prompt_id,
                "bedrock_prompt_arn": prompt_arn,
                "bedrock_version": bedrock_prompt_version,
            }
        except Exception as exc:
            result = {"status": "error", "error": str(exc)}

        self._sync_cache[cache_key] = result
        return result

    def sync_all_templates(self) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        for template_name, cfg in self.registry.get("templates", {}).items():
            active_version = cfg.get("active_version", "v1")
            results[template_name] = self.sync_template_to_bedrock(
                template_name, active_version
            )
        return results


if __name__ == "__main__":
    manager = BedrockPromptManager()
    print(json.dumps(manager.sync_all_templates(), indent=2))
