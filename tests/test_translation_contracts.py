import dataclasses
import unittest

from crimson0020.translation_runtime.contracts import (
    FailureKind,
    RouteAttempt,
    TermConstraint,
    TermSuggestion,
    TranslationBatchOutcome,
    TranslationBatchRequest,
    TranslationFailure,
    TranslationItemResult,
    TranslationRequestItem,
)
from crimson0020.translation_runtime.errors import (
    ContractValidationError,
    TranslationRuntimeError,
)


class TestTranslationContracts(unittest.TestCase):
    def test_term_constraint_immutability(self):
        self.assertTrue(issubclass(ContractValidationError, TranslationRuntimeError))
        term = TermConstraint(source_term="Dragon", target_term="Rồng")
        self.assertEqual(term.source_term, "Dragon")
        self.assertEqual(term.target_term, "Rồng")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            term.source_term = "New"

    def test_term_suggestion_immutability_and_confidence(self):
        sug = TermSuggestion(source_term="Sword", suggested_target="Kiếm", confidence=0.95)
        self.assertEqual(sug.confidence, 0.95)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            sug.confidence = 0.5

    def test_request_item_immutability_and_fields(self):
        item = TranslationRequestItem(
            work_id="w-001",
            masked_text="Hello <x0>",
            context=("line 1", "line 2"),
            context_signature="sig-123",
            hard_terms=(TermConstraint("Dragon", "Rồng"),),
            soft_terms=(TermSuggestion("Sword", "Kiếm", 0.9),),
        )
        self.assertEqual(item.work_id, "w-001")
        self.assertEqual(item.masked_text, "Hello <x0>")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            item.work_id = "w-002"

        # Check absence of sensitive routing identity (marker, key, path)
        fields = {f.name for f in dataclasses.fields(item)}
        self.assertNotIn("marker", fields)
        self.assertNotIn("key", fields)
        self.assertNotIn("path", fields)

    def test_batch_request_validation(self):
        item = TranslationRequestItem(
            work_id="w-001",
            masked_text="Hello",
            context=(),
            context_signature="sig",
            hard_terms=(),
            soft_terms=(),
        )
        batch = TranslationBatchRequest(
            job_id="job-001",
            source_language="en",
            target_language="vi",
            items=(item,),
        )
        self.assertEqual(batch.job_id, "job-001")
        self.assertEqual(len(batch.items), 1)

        # Empty language or job_id must raise validation error
        with self.assertRaises(ContractValidationError):
            TranslationBatchRequest(
                job_id="",
                source_language="en",
                target_language="vi",
                items=(item,),
            )
        with self.assertRaises(ContractValidationError):
            TranslationBatchRequest(
                job_id="job-1",
                source_language="",
                target_language="vi",
                items=(item,),
            )

    def test_failure_kinds_complete(self):
        expected_kinds = {
            "network",
            "timeout",
            "rate_limit",
            "unauthorized",
            "unsupported_model",
            "invalid_structure",
            "missing_item",
            "duplicate_item",
            "empty_translation",
            "protected_content_violation",
            "glossary_violation",
            "cancellation",
            "provider_internal",
        }
        actual_kinds = {k.value for k in FailureKind}
        self.assertEqual(actual_kinds, expected_kinds)

    def test_translation_failure_immutability(self):
        fail = TranslationFailure(
            work_id="w-001",
            kind=FailureKind.RATE_LIMIT,
            retryable=True,
            message="Too many requests",
            retry_after_seconds=30.0,
        )
        self.assertEqual(fail.kind, FailureKind.RATE_LIMIT)
        self.assertTrue(fail.retryable)
        self.assertEqual(fail.retry_after_seconds, 30.0)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            fail.retryable = False

    def test_batch_outcome(self):
        res = TranslationItemResult(
            work_id="w-001",
            translated_text="Xin chào",
            route_id="route-opencode",
            model_id="model-free",
        )
        attempt = RouteAttempt(
            route_id="route-opencode",
            model_id="model-free",
            duration_ms=250.0,
        )
        outcome = TranslationBatchOutcome(
            successes=(res,),
            failures=(),
            attempts=(attempt,),
        )
        self.assertEqual(len(outcome.successes), 1)
        self.assertEqual(outcome.successes[0].translated_text, "Xin chào")
        self.assertEqual(len(outcome.failures), 0)
        self.assertEqual(len(outcome.attempts), 1)

    def test_serialization_roundtrip(self):
        item = TranslationRequestItem(
            work_id="w-001",
            masked_text="Hello",
            context=("ctx1",),
            context_signature="sig1",
            hard_terms=(TermConstraint("Dragon", "Rồng"),),
            soft_terms=(TermSuggestion("Sword", "Kiếm", 0.9),),
        )
        batch = TranslationBatchRequest(
            job_id="job-001",
            source_language="en",
            target_language="vi",
            items=(item,),
        )
        d = batch.to_dict()
        restored = TranslationBatchRequest.from_dict(d)
        self.assertEqual(batch, restored)

    def test_equality_and_hash(self):
        t1 = TermConstraint("A", "B")
        t2 = TermConstraint("A", "B")
        self.assertEqual(t1, t2)
        self.assertEqual(hash(t1), hash(t2))
        s = {t1}
        self.assertIn(t2, s)

    def test_term_constraint_validation(self):
        with self.assertRaises(ContractValidationError):
            TermConstraint("", "Target")
        with self.assertRaises(ContractValidationError):
            TermConstraint("Source", "")
        t = TermConstraint("A", "B")
        self.assertEqual(TermConstraint.from_dict(t.to_dict()), t)

    def test_term_suggestion_validation(self):
        with self.assertRaises(ContractValidationError):
            TermSuggestion("", "Target")
        with self.assertRaises(ContractValidationError):
            TermSuggestion("Source", "")
        with self.assertRaises(ContractValidationError):
            TermSuggestion("Source", "Target", confidence=1.5)
        with self.assertRaises(ContractValidationError):
            TermSuggestion("Source", "Target", confidence=-0.1)
        s = TermSuggestion("A", "B", 0.8)
        self.assertEqual(TermSuggestion.from_dict(s.to_dict()), s)

    def test_translation_request_item_validation(self):
        with self.assertRaises(ContractValidationError):
            TranslationRequestItem("", "text", (), "sig", (), ())
        with self.assertRaises(ContractValidationError):
            TranslationRequestItem("w1", 123, (), "sig", (), ())  # type: ignore
        with self.assertRaises(ContractValidationError):
            TranslationRequestItem("w1", "text", (), 123, (), ())  # type: ignore

        # Lists converted to tuples
        item = TranslationRequestItem("w1", "text", ["c1"], "sig", [TermConstraint("A", "B")], [TermSuggestion("C", "D")])  # type: ignore
        self.assertIsInstance(item.context, tuple)
        self.assertIsInstance(item.hard_terms, tuple)
        self.assertIsInstance(item.soft_terms, tuple)
        self.assertEqual(TranslationRequestItem.from_dict(item.to_dict()), item)

    def test_translation_item_result_validation(self):
        with self.assertRaises(ContractValidationError):
            TranslationItemResult("", "text", "r1", "m1")
        with self.assertRaises(ContractValidationError):
            TranslationItemResult("w1", 123, "r1", "m1")  # type: ignore
        with self.assertRaises(ContractValidationError):
            TranslationItemResult("w1", "text", 123, "m1")  # type: ignore
        with self.assertRaises(ContractValidationError):
            TranslationItemResult("w1", "text", "r1", 123)  # type: ignore
        res = TranslationItemResult("w1", "translated", "r1", "m1")
        self.assertEqual(TranslationItemResult.from_dict(res.to_dict()), res)

    def test_translation_failure_validation(self):
        with self.assertRaises(ContractValidationError):
            TranslationFailure("", FailureKind.TIMEOUT, True, "msg")
        with self.assertRaises(ContractValidationError):
            TranslationFailure("w1", 123, True, "msg")  # type: ignore
        with self.assertRaises(ContractValidationError):
            TranslationFailure("w1", FailureKind.TIMEOUT, True, 123)  # type: ignore
        with self.assertRaises(ContractValidationError):
            TranslationFailure("w1", FailureKind.TIMEOUT, True, "msg", retry_after_seconds=-5.0)

        # String kind converted to enum
        fail = TranslationFailure("w1", "timeout", True, "msg", 10.0)  # type: ignore
        self.assertEqual(fail.kind, FailureKind.TIMEOUT)
        self.assertEqual(TranslationFailure.from_dict(fail.to_dict()), fail)

    def test_route_attempt_validation(self):
        with self.assertRaises(ContractValidationError):
            RouteAttempt(123, "m1")  # type: ignore
        with self.assertRaises(ContractValidationError):
            RouteAttempt("r1", 123)  # type: ignore
        with self.assertRaises(ContractValidationError):
            RouteAttempt("r1", "m1", error_kind=123)  # type: ignore

        attempt = RouteAttempt("r1", "m1", 150.0, "rate_limit", 429)  # type: ignore
        self.assertEqual(attempt.error_kind, FailureKind.RATE_LIMIT)
        self.assertEqual(RouteAttempt.from_dict(attempt.to_dict()), attempt)

    def test_translation_batch_outcome_roundtrip(self):
        outcome = TranslationBatchOutcome(
            successes=[TranslationItemResult("w1", "trans", "r1", "m1")],  # type: ignore
            failures=[TranslationFailure("w2", FailureKind.TIMEOUT, True, "msg", 5.0)],  # type: ignore
            attempts=[RouteAttempt("r1", "m1", 100.0)],  # type: ignore
        )
        self.assertIsInstance(outcome.successes, tuple)
        self.assertIsInstance(outcome.failures, tuple)
        self.assertIsInstance(outcome.attempts, tuple)
        d = outcome.to_dict()
        restored = TranslationBatchOutcome.from_dict(d)
        self.assertEqual(outcome, restored)


if __name__ == "__main__":
    unittest.main()
