"""
auto_rd.py
==========
Automatic "Research & Decision" ASSISTANT for SafeFrame access requests.

IMPORTANT: this engine never makes the final call. It only investigates
a customer the instant they submit a request and hands the admin a
ready-made report — trust score, risk level, a recommendation, and a
plain-English summary. The admin always clicks Approve or Deny.

Score: 0-100, built from 5 weighted signals (each worth up to 20 points).
  70-100 -> LOW risk    -> recommend APPROVE
  40-69  -> MEDIUM risk -> recommend REVIEW
  0-39   -> HIGH risk   -> recommend REJECT
"""

from datetime import datetime, timezone

BUSINESS_KEYWORDS    = ["retailer", "brand", "business", "campaign", "marketing",
                        "partner", "agency", "client", "wholesale", "distributor"]
SUSPICIOUS_KEYWORDS  = ["personal", "curious", "just want", "bored", "fun", "random"]
FREE_EMAIL_DOMAINS   = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com"]

LOW_RISK_AT    = 70
MEDIUM_RISK_AT = 40


def _account_age_days(created_at: str) -> int:
    # WHAT: how many days ago was this account created
    # WHY:  older accounts are statistically more trustworthy
    # IN:   created_at (ISO timestamp string)
    # OUT:  integer number of days (0 if it can't be parsed)
    try:
        created = datetime.fromisoformat(created_at)
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - created).days
    except Exception:
        return 0


def score_account_age(created_at: str) -> int:
    days = _account_age_days(created_at)
    if days >= 30:
        return 20
    if days >= 15:
        return 10
    return 5


def score_request_history(past_request_count: int) -> int:
    if past_request_count <= 2:
        return 20
    if past_request_count <= 7:
        return 15
    if past_request_count <= 15:
        return 5
    return 0


def score_denial_history(past_denials: int) -> int:
    if past_denials == 0:
        return 20
    if past_denials == 1:
        return 15
    if past_denials == 2:
        return 5
    return 0


def score_reason_quality(reason: str) -> int:
    text  = (reason or "").lower()
    words = text.split()
    score = 0
    if len(words) > 20:
        score += 10
    if any(kw in text for kw in BUSINESS_KEYWORDS):
        score += 10
    if any(kw in text for kw in SUSPICIOUS_KEYWORDS):
        score -= 10
    return max(0, min(20, score))


def score_email_domain(email: str, created_at: str) -> int:
    domain = email.split("@")[-1].lower().strip()
    if domain not in FREE_EMAIL_DOMAINS:
        return 20
    if _account_age_days(created_at) >= 30:
        return 10
    return 5


def _risk_level(total: int) -> str:
    # WHAT: translate the numeric score into a LOW/MEDIUM/HIGH risk label
    # WHY:  the admin needs an at-a-glance read before diving into the breakdown
    # IN:   total trust score (0-100)
    # OUT:  "LOW" | "MEDIUM" | "HIGH"
    if total >= LOW_RISK_AT:
        return "LOW"
    if total >= MEDIUM_RISK_AT:
        return "MEDIUM"
    return "HIGH"


def _recommendation(risk: str) -> str:
    # WHAT: map a risk level onto a one-word suggestion for the admin
    # WHY:  gives the admin a starting point — they still decide either way
    # IN:   risk level string
    # OUT:  "APPROVE" | "REVIEW" | "REJECT"
    return {"LOW": "APPROVE", "MEDIUM": "REVIEW", "HIGH": "REJECT"}[risk]


def _summary(customer: dict, reason: str, breakdown: dict, risk: str,
             past_request_count: int, past_denials: int) -> str:
    # WHAT: generate a short, plain-English read-out of why the score landed where it did
    # WHY:  admins shouldn't have to mentally re-derive the story behind 5 numbers
    # IN:   customer dict, reason text, score breakdown, risk level, history counts
    # OUT:  a short human-readable paragraph
    domain      = customer.get("email", "").split("@")[-1].lower()
    is_business = domain not in FREE_EMAIL_DOMAINS
    text        = (reason or "").lower()
    has_biz_kw  = any(kw in text for kw in BUSINESS_KEYWORDS)
    has_sus_kw  = any(kw in text for kw in SUSPICIOUS_KEYWORDS)
    detailed    = len((reason or "").split()) > 20

    bits = []
    bits.append("a business email" if is_business else "a free webmail address")
    bits.append("a clean request history" if past_denials == 0 else
                f"{past_denials} previous denial{'s' if past_denials != 1 else ''} on record")
    if detailed and has_biz_kw:
        bits.append("a detailed, business-oriented reason")
    elif has_sus_kw:
        bits.append("vague or suspicious wording in their reason")
    elif detailed:
        bits.append("a reasonably detailed reason")
    else:
        bits.append("a short, low-detail reason")

    profile = "This customer has " + ", ".join(bits) + "."

    if risk == "LOW":
        verdict = "Low risk profile — recommended for approval."
    elif risk == "MEDIUM":
        verdict = "Medium risk — worth a closer manual look before deciding."
    else:
        verdict = "High risk — recommend rejection unless there's strong context to the contrary."

    return f"{profile} {verdict}"


def evaluate_request(customer: dict, reason: str,
                      past_request_count: int, past_denials: int) -> dict:
    # WHAT: investigate a request and produce a full report — score, risk,
    #       recommendation and summary — WITHOUT deciding anything
    # WHY:  this is the single entry point app.py calls for every new request;
    #       the admin always makes the final approve/deny call
    # IN:   customer (user dict), reason (string), past_request_count, past_denials
    # OUT:  dict with breakdown, total score, risk_level, recommendation, summary
    breakdown = {
        "account_age":     score_account_age(customer.get("created_at", "")),
        "request_history": score_request_history(past_request_count),
        "denial_history":  score_denial_history(past_denials),
        "reason_quality":  score_reason_quality(reason),
        "email_domain":    score_email_domain(customer.get("email", ""), customer.get("created_at", "")),
    }
    total = sum(breakdown.values())
    risk  = _risk_level(total)
    rec   = _recommendation(risk)
    summary = _summary(customer, reason, breakdown, risk, past_request_count, past_denials)

    return {
        "breakdown":      breakdown,
        "total":          total,
        "risk_level":     risk,
        "recommendation": rec,
        "summary":        summary,
    }
