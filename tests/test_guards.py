import pytest

from faqcache.guards import TTLPolicy, find_user_data, has_negation, negation_mismatch, redact


@pytest.mark.parametrize(
    "text",
    ["refund nahi milega", "I have not received it", "Why can't I cancel?", "Why can’t I pay?",
     "order mat bhejo", "never arrived", "It doesn't work", "return without the box"],
)  # fmt: skip
def test_negation_detected(text):
    assert has_negation(text)


@pytest.mark.parametrize(
    "text", ["refund milega?", "Do you offer no-cost EMI?", "No cost EMI milta hai kya?",
             "How do I track my order?", "Is the format right?"],
)  # fmt: skip
def test_no_negation(text):
    assert not has_negation(text)


def test_negation_mismatch():
    assert negation_mismatch("refund milega", "refund nahi milega")
    assert not negation_mismatch("refund nahi aaya", "I have not received my refund")
    assert not negation_mismatch("refund kab milega", "how long does a refund take")


@pytest.mark.parametrize(
    "text,kind",
    [("Where is order ORD-123456?", "order_id"), ("ticket #98765 is late", "ref_number"),
     ("call me on 9876543210", "phone"), ("mail a.b@example.com", "email"),
     ("My name is Riya", "name"), ("Hi Rahul, your refund is done", "greeting_name")],
)  # fmt: skip
def test_user_data_found(text, kind):
    assert find_user_data(text) == kind


def test_user_data_absent_in_normal_faq():
    assert find_user_data("Refunds take 5-7 working days. Call 1800-000-0000.") is None


def test_redact_masks_personal_data():
    out = redact("order ORD-123456, phone 9876543210")
    assert "123456" not in out and "9876543210" not in out


def test_ttl_policy():
    p = TTLPolicy(volatile_ttl_s=100, default_ttl_s=None)
    assert p.ttl_for("What are the delivery charges?", "Rs 40") == 100
    assert p.ttl_for("How do I reset my password?", "Use Forgot password") is None
