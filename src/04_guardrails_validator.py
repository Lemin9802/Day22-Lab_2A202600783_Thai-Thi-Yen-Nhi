"""
Bước 4 - Guardrails Validator
=============================
  1. Tạo custom PII validator:
     - detect email
     - detect phone number
     - detect SSN-like number
     - detect credit-card-like number
     - redact bằng on_fail=OnFailAction.FIX

  2. Tạo custom JSON formatter validator:
     - remove markdown fences
     - repair single quotes
     - remove trailing commas
     - fallback JSON nếu không parse được

  3. Lưu evidence:
     - evidence/04_pii_demo_log.txt
     - evidence/04_json_demo_log.txt
"""
import ast
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).parent))

from guardrails import Guard, OnFailAction
from guardrails.validator_base import Validator, PassResult, FailResult
from guardrails.validators import register_validator


# ── 1. PII Validator ───────────────────────────────────────────────────────
@register_validator(name="pii_detector", data_type="string")
class PIIDetector(Validator):
    """
    Detect và redact PII trong text.

    Detect ít nhất 3 loại PII:
      - email
      - phone number
      - SSN-like number
      - credit-card-like number

    on_fail được truyền vào constructor:
      PIIDetector(on_fail=OnFailAction.FIX)
    """

    EMAIL_RE = re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    )

    PHONE_RE = re.compile(
        r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{3,4}(?!\d)"
    )

    SSN_RE = re.compile(
        r"\b\d{3}-\d{2}-\d{4}\b"
    )

    CREDIT_CARD_RE = re.compile(
        r"\b(?:\d[ -]*?){13,16}\b"
    )

    def _find_pii(self, text: str) -> Dict[str, list[str]]:
        matches = {
            "email": self.EMAIL_RE.findall(text),
            "phone": self.PHONE_RE.findall(text),
            "ssn": self.SSN_RE.findall(text),
            "credit_card": [],
        }

        # Lọc credit card để tránh match phone quá ngắn.
        for match in self.CREDIT_CARD_RE.findall(text):
            digits = re.sub(r"\D", "", match)
            if 13 <= len(digits) <= 16:
                matches["credit_card"].append(match)

        return {key: values for key, values in matches.items() if values}

    def _redact(self, text: str) -> str:
        redacted = text

        redacted = self.EMAIL_RE.sub("[REDACTED_EMAIL]", redacted)
        redacted = self.SSN_RE.sub("[REDACTED_SSN]", redacted)
        redacted = self.CREDIT_CARD_RE.sub("[REDACTED_CARD]", redacted)
        redacted = self.PHONE_RE.sub("[REDACTED_PHONE]", redacted)

        return redacted

    def validate(self, value: Any, metadata: Dict[str, Any]) -> PassResult | FailResult:
        text = str(value)
        matches = self._find_pii(text)

        if not matches:
            return PassResult()

        return FailResult(
            error_message=f"PII detected: {', '.join(matches.keys())}",
            fix_value=self._redact(text),
        )


# ── 2. JSON Formatter Validator ────────────────────────────────────────────
@register_validator(name="json_formatter", data_type="string")
class JSONFormatter(Validator):
    """
    Repair text thành JSON string hợp lệ.

    Hỗ trợ:
      - remove markdown fences
      - convert Python/single-quote dict bằng ast.literal_eval
      - remove trailing commas
      - fallback JSON nếu không parse được
    """

    def _strip_markdown_fences(self, text: str) -> str:
        cleaned = text.strip()

        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        )

        return cleaned.strip()

    def _remove_trailing_commas(self, text: str) -> str:
        return re.sub(r",\s*([}\]])", r"\1", text)

    def _fallback_json(self, raw_text: str) -> str:
        return json.dumps(
            {
                "error": "Could not parse JSON",
                "raw_output": raw_text,
            },
            ensure_ascii=False,
        )

    def _repair_json(self, text: str) -> tuple[bool, str]:
        original = str(text)
        cleaned = self._strip_markdown_fences(original)
        cleaned = self._remove_trailing_commas(cleaned)

        # Case 1: already valid JSON after cleaning.
        try:
            parsed = json.loads(cleaned)
            return True, json.dumps(parsed, ensure_ascii=False)
        except Exception:
            pass

        # Case 2: Python-style dict/list with single quotes.
        try:
            parsed = ast.literal_eval(cleaned)
            return True, json.dumps(parsed, ensure_ascii=False)
        except Exception:
            pass

        # Case 3: very small repair for simple single-quote JSON-like strings.
        try:
            quote_fixed = re.sub(r"'", '"', cleaned)
            quote_fixed = self._remove_trailing_commas(quote_fixed)
            parsed = json.loads(quote_fixed)
            return True, json.dumps(parsed, ensure_ascii=False)
        except Exception:
            pass

        # Case 4: fallback JSON.
        return False, self._fallback_json(original)

    def validate(self, value: Any, metadata: Dict[str, Any]) -> PassResult | FailResult:
        original = str(value)
        ok, fixed = self._repair_json(original)

        try:
            json.loads(original)
            original_valid = True
        except Exception:
            original_valid = False

        if original_valid:
            return PassResult()

        if ok:
            return FailResult(
                error_message="Invalid JSON was repaired.",
                fix_value=fixed,
            )

        return FailResult(
            error_message="Invalid JSON could not be repaired. Returning fallback JSON.",
            fix_value=fixed,
        )


# ── 3. Helper functions ────────────────────────────────────────────────────
def get_validated_output(outcome):
    """
    Guardrails versions expose validated output with slightly different attribute names.
    This helper keeps the demo robust.
    """
    if hasattr(outcome, "validated_output"):
        return outcome.validated_output

    if hasattr(outcome, "raw_llm_output"):
        return outcome.raw_llm_output

    return outcome


def get_validation_passed(outcome) -> bool:
    if hasattr(outcome, "validation_passed"):
        return bool(outcome.validation_passed)

    if hasattr(outcome, "passed"):
        return bool(outcome.passed)

    return False


def run_guard(guard: Guard, value: str):
    outcome = guard.validate(value)
    return {
        "passed": get_validation_passed(outcome),
        "validated_output": get_validated_output(outcome),
        "raw_outcome_type": type(outcome).__name__,
    }


# ── 4. Demos ───────────────────────────────────────────────────────────────
def run_pii_demo():
    pii_guard = Guard().use(
        PIIDetector(on_fail=OnFailAction.FIX)
    )

    examples = [
        "Contact Alice at alice@example.com for details.",
        "Call me at +1 415-555-2671 tomorrow.",
        "Customer SSN is 123-45-6789 and card is 4111 1111 1111 1111.",
        "This sentence has no private information.",
    ]

    results = []

    for example in examples:
        result = run_guard(pii_guard, example)
        results.append(
            {
                "input": example,
                "passed": result["passed"],
                "output": str(result["validated_output"]),
                "outcome_type": result["raw_outcome_type"],
            }
        )

    return results


def run_json_demo():
    json_guard = Guard().use(
        JSONFormatter(on_fail=OnFailAction.FIX)
    )

    examples = [
        """```json
{"name": "Alice", "role": "engineer",}
```""",
        "{'name': 'Bob', 'score': 0.91,}",
        '{"valid": true, "items": [1, 2, 3]}',
        "not json at all",
    ]

    results = []

    for example in examples:
        result = run_guard(json_guard, example)
        output = str(result["validated_output"])

        # Verify output is valid JSON.
        try:
            json.loads(output)
            output_is_valid_json = True
        except Exception:
            output_is_valid_json = False

        results.append(
            {
                "input": example,
                "passed": result["passed"],
                "output": output,
                "output_is_valid_json": output_is_valid_json,
                "outcome_type": result["raw_outcome_type"],
            }
        )

    return results


def write_pii_log(results: list[dict]):
    evidence_dir = Path(__file__).parent.parent / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    log_path = evidence_dir / "04_pii_demo_log.txt"

    with log_path.open("w", encoding="utf-8") as f:
        f.write("Day22 Task 4 - PII Guardrails Demo\n")
        f.write("=" * 72 + "\n")
        f.write(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write("Validator: PIIDetector(on_fail=OnFailAction.FIX)\n")
        f.write("Detected PII types: email, phone, SSN, credit-card-like number\n")
        f.write("=" * 72 + "\n\n")

        for i, item in enumerate(results, start=1):
            f.write(f"[{i}] Input:\n{item['input']}\n")
            f.write(f"Passed: {item['passed']}\n")
            f.write(f"Outcome type: {item['outcome_type']}\n")
            f.write(f"Output:\n{item['output']}\n")
            f.write("-" * 72 + "\n")

    return log_path


def write_json_log(results: list[dict]):
    evidence_dir = Path(__file__).parent.parent / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    log_path = evidence_dir / "04_json_demo_log.txt"

    with log_path.open("w", encoding="utf-8") as f:
        f.write("Day22 Task 4 - JSON Guardrails Demo\n")
        f.write("=" * 72 + "\n")
        f.write(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write("Validator: JSONFormatter(on_fail=OnFailAction.FIX)\n")
        f.write("Repairs: markdown fences, single quotes, trailing commas, fallback JSON\n")
        f.write("=" * 72 + "\n\n")

        for i, item in enumerate(results, start=1):
            f.write(f"[{i}] Input:\n{item['input']}\n")
            f.write(f"Passed: {item['passed']}\n")
            f.write(f"Output is valid JSON: {item['output_is_valid_json']}\n")
            f.write(f"Outcome type: {item['outcome_type']}\n")
            f.write(f"Output:\n{item['output']}\n")
            f.write("-" * 72 + "\n")

    return log_path


def main():
    print("=" * 60)
    print("  Bước 4: Guardrails Validator")
    print("=" * 60)

    pii_results = run_pii_demo()
    json_results = run_json_demo()

    pii_log = write_pii_log(pii_results)
    json_log = write_json_log(json_results)

    print("\nPII demo results:")
    for i, item in enumerate(pii_results, start=1):
        print(f"[{i}] passed={item['passed']} output={item['output']}")

    print("\nJSON demo results:")
    for i, item in enumerate(json_results, start=1):
        print(
            f"[{i}] passed={item['passed']} "
            f"valid_json={item['output_is_valid_json']} "
            f"output={item['output']}"
        )

    print("\nSaved evidence:")
    print(f"  {pii_log}")
    print(f"  {json_log}")
    print("\nTask 4 completed.")


if __name__ == "__main__":
    main()