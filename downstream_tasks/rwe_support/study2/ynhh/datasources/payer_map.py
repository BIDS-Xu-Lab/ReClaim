# =============================================================================
# payer_concept_id -> payer class for the Table S5 insurance covariate (ins_category).
# Classes: CCAE = commercial, MDCR = Medicare, MDCD = Medicaid, MISSING = other/unknown,
# so the covariate has the same meaning as in a claims source. Periods are keyed on START
# dates only (see OMOPAdapter.insurance_at_index).
# =============================================================================
PAYER_CONCEPT_CLASS = {
       0: "MISSING",    # No matching concept
     280: "MDCR",       # Medicare
     281: "MDCR",       # Medicare Managed Care
     283: "MDCR",       # Medicare Preferred Provider Organization (PPO)
     289: "MDCD",       # Medicaid
     290: "MDCD",       # Medicaid Managed Care
     295: "MISSING",    # Other Government Federal/State/Local, excluding Department of Corrections
     296: "CCAE",       # DoD TRICARE, Civilian Health and Medical Program of the Uniformed Services (CHAMPUS)
     325: "MISSING",    # Department of Corrections
     327: "CCAE",       # Private Health Insurance
     328: "CCAE",       # Private Health Insurance - Managed Care
     330: "CCAE",       # Commercial Managed Care - Preferred Provider Organization (PPO)
     331: "CCAE",       # Commercial Managed Care - Point of Service (POS)
     332: "CCAE",       # Blue Cross/Blue Shield
     333: "CCAE",       # Blue Cross Managed Care - Health Management Organization (HMO)
     334: "CCAE",       # Blue Cross Managed Care - Preferred Provider Organization (PPO)
     336: "CCAE",       # Blue Cross Managed Care - Dental
     337: "CCAE",       # Blue Cross Insurance Indemnity
     340: "CCAE",       # Managed Care, Unspecified
     341: "CCAE",       # Managed Care Health Management Organization (HMO)
     342: "CCAE",       # Managed Care Preferred Provider Organization (PPO)
     344: "MISSING",    # Miscellaneous Program
     345: "CCAE",       # Other Non-government
     348: "CCAE",       # Worker's Comp Fee-for-Service (FFS)
     365: "MDCR",       # Medicare Other
     374: "MDCD",       # Medicaid Dental
     376: "MDCR",       # DoD TRICARE For Life - Medicare Supplement
     382: "MISSING",    # DoD Dental - Stand Alone
     383: "CCAE",       # Department of Veterans Affairs
     392: "CCAE",       # VA Civilian Health and Medical Program for the VA (CHAMPVA)
     408: "MISSING",    # Federal, State, Local not specified - Other
     412: "CCAE",       # Exclusive Provider Organization
     415: "CCAE",       # Commercial Managed Care - Dental
     416: "CCAE",       # Managed Care, Other (non HMO)
     418: "CCAE",       # Commercial Indemnity
     420: "MDCR",       # Medicare supplemental policy (as second payer)
     427: "MISSING",    # Dental Other Private Insurance
     429: "CCAE",       # Blue Cross Managed Care
     430: "CCAE",       # Blue Cross Managed Care - Other
     433: "CCAE",       # Managed Care Point of Service (POS)
     434: "CCAE",       # Other Managed Care
     435: "MISSING",    # No Payment from an Organization/Agency/Program/Private Payer Listed
     436: "MISSING",    # Self-pay
     447: "CCAE",       # Worker's Compensation
}
PAYER_CLASS_LABEL = {"CCAE": "commercial", "MDCR": "medicare",
                     "MDCD": "medicaid", "MISSING": "other_unknown"}


def classify_payer_concept(concept_id) -> str:
    """payer_concept_id -> commercial|medicare|medicaid|other_unknown (unmapped -> other_unknown)."""
    try:
        cid = int(concept_id)
    except (TypeError, ValueError):
        return "other_unknown"
    return PAYER_CLASS_LABEL[PAYER_CONCEPT_CLASS.get(cid, "MISSING")]
