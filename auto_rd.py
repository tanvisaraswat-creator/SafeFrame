"""
auto_rd.py
==========
Fully automatic "Research & Decision" engine for SafeFrame access requests.
No human needed for the vast majority of cases — it scores every customer
request the instant it's submitted and returns an instant decision.

Score: 0-100, built from 5 weighted signals (each worth up to 20 points).
  70-100 -> AUTO_APPROVED
  40-69  -> PENDING (sent to admin for manual review)
  0-39   -> AUTO_DENIED
"""

from datetime import datetime, timezone

BUSINESS_KEYWORDS    = ["retailer", "brand", "business", "campaign", "marketing",
                        "partner", "agency", "client", "wholesale", "distributor"]
SUSPICIOUS_KEYWORDS  = ["personal", "curious", "just want", "bored", "fun", "random"]
FREE_EMAIL_DOMAINS   = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com"]

AUTO_APPROVE_AT = 70
AUTO_DENY_BELOW = 40


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
    # WHAT: award 0-20 points based on how long the account has existed
    # WHY:  brand-new accounts asking for access are higher risk
    # IN:   created_at ISO string
    # OUT:  integer score 0-20
    days = _account_age_days(created_at)
    if days >= 30:
        return 20
    if days >= 15:
        return 10
    return 5


def score_request_history(past_request_count: int) -> int:
    # WHAT: award 0-20 points based on how many requests this customer has made
    # WHY:  a flood of requests is a sign of scraping / abuse, not genuine interest
    # IN:   past_request_count (int, requests made before this one)
    # OUT:  integer score 0-20
    if past_request_count <= 2:
        return 20
    if past_request_count <= 7:
        return 15
    if past_request_count <= 15:
        return 5
    return 0


def score_denial_history(past_denials: int) -> int:
    # WHAT: award 0-20 points based on how many past requests were denied
    # WHY:  repeated denials are a strong signal the customer shouldn't be trusted
    # IN:   past_denials (int)
    # OUT:  integer score 0-20
    if past_denials == 0:
        return 20
    if past_denials == 1:
        return 15
    if past_denials == 2:
        return 5
    return 0


def score_reason_quality(reason: str) -> int:
    # WHAT: award up to 20 points for a thoughtful, business-sounding reason
    # WHY:  the explanation a customer gives is the clearest signal of genuine intent
    # IN:   reason (free-text string the customer typed)
    # OUT:  integer score, clamped to 0-20
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
    # WHAT: award 0-20 points based on whether the email looks like a business address
    # WHY:  business domains are harder to fake than free webmail accounts
    # IN:   email string, created_at ISO string (to judge "old" free-mail accounts)
    # OUT:  integer score 0-20
    domain = email.split("@")[-1].lower().strip()
    if domain not in FREE_EMAIL_DOMAINS:
        return 20
    if _account_age_days(created_at) >= 30:
        return 10
    return 5


def evaluate_request(customer: dict, reason: str,
                      past_request_count: int, past_denials: int) -> dict:
    # WHAT: run all 5 scoring signals and produce a final trust score + decision
    # WHY:  this is the single entry point app.py calls for every new request
    # IN:   customer (user dict), reason (string), past_request_count, past_denials
    # OUT:  dict with breakdown, total score, decision, and a human-readable message
    breakdown = {
        "account_age":     score_account_age(customer.get("created_at", "")),
        "request_history": score_request_history(past_request_count),
        "denial_history":  score_denial_history(past_denials),
        "reason_quality":  score_reason_quality(reason),
        "email_domain":    score_email_domain(customer.get("email", ""), customer.get("created_at", "")),
    }
    total = sum(breakdown.values())

    if total >= AUTO_APPROVE_AT:
        decision = "auto_approved"
        message  = f"Auto-approved! Trust Score: {total}/100"
    elif total >= AUTO_DENY_BELOW:
        decision = "pending"
        message  = f"Under review. Trust Score: {total}/100 — Admin will review within 24 hours"
    else:
        decision = "auto_denied"
        message  = f"Access denied. Trust Score: {total}/100 — Reason: Insufficient business verification"

    return {
        "breakdown": breakdown,
        "total":     total,
        "decision":  decision,
        "message":   message,
    }
