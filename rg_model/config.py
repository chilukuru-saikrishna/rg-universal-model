"""Shared configuration: the 89 responsible gambling (RG) data points and global settings."""

RANDOM_STATE = 42

# RG id -> (data point name, category). Matches RG_Data_Points_iGaming.xlsx.
META = {1: ('Age', '1. Player profile'), 2: ('Gender', '1. Player profile'), 3: ('Country / jurisdiction', '1. Player profile'), 4: ('Account age', '1. Player profile'), 5: ('Income / occupation band', '1. Player profile'), 6: ('VIP / high-value status', '1. Player profile'), 7: ('Active days', '2. Frequency & time'), 8: ('Number of sessions', '2. Frequency & time'), 9: ('Average session length', '2. Frequency & time'), 10: ('Maximum session length', '2. Frequency & time'), 11: ('Total time gambling', '2. Frequency & time'), 12: ('Night-time play', '2. Frequency & time'), 13: ('Weekend play', '2. Frequency & time'), 14: ('Play during working hours', '2. Frequency & time'), 15: ('Continuous play without breaks', '2. Frequency & time'), 16: ('Consecutive days played', '2. Frequency & time'), 17: ('Time-of-day shift', '2. Frequency & time'), 18: ('Deposit count', '3. Deposits'), 19: ('Deposit amount', '3. Deposits'), 20: ('Deposits per session', '3. Deposits'), 21: ('Multiple deposits in short time', '3. Deposits'), 22: ('Deposit soon after a loss', '3. Deposits'), 23: ('Time between deposits', '3. Deposits'), 24: ('Deposit escalation', '3. Deposits'), 25: ('Failed / declined deposits', '3. Deposits'), 26: ('Night-time deposits', '3. Deposits'), 27: ('Deposit limit usage', '3. Deposits'), 28: ('Withdrawal count & amount', '4. Withdrawals'), 29: ('Withdrawal variability', '4. Withdrawals'), 30: ('Cancelled / reversed withdrawals', '4. Withdrawals'), 31: ('Withdrawal-to-deposit ratio', '4. Withdrawals'), 32: ('Withdraw then redeposit', '4. Withdrawals'), 33: ('Number of bets', '5. Betting & staking'), 34: ('Total stake (turnover)', '5. Betting & staking'), 35: ('Average stake per bet', '5. Betting & staking'), 36: ('Maximum single bet / daily stake', '5. Betting & staking'), 37: ('Stake variability', '5. Betting & staking'), 38: ('Stake escalation', '5. Betting & staking'), 39: ('Bet speed', '5. Betting & staking'), 40: ('Stake increase after losses', '5. Betting & staking'), 41: ('Net loss', '6. Losses & financial outcome'), 42: ('Loss per session / per active day', '6. Losses & financial outcome'), 43: ('Loss ratio', '6. Losses & financial outcome'), 44: ('Loss escalation', '6. Losses & financial outcome'), 45: ('Account depletion', '6. Losses & financial outcome'), 46: ('Large single-day loss', '6. Losses & financial outcome'), 47: ('Lifetime net loss', '6. Losses & financial outcome'), 48: ('Loss vs affordability', '6. Losses & financial outcome'), 49: ('Number of product types', '7. Products & games'), 50: ('Share of high-risk products', '7. Products & games'), 51: ('Game volatility', '7. Products & games'), 52: ('Game switching', '7. Products & games'), 53: ('In-play betting share', '7. Products & games'), 54: ('Product migration', '7. Products & games'), 55: ('Limits set', '8. RG tool use'), 56: ('Limit increases', '8. RG tool use'), 57: ('Limit removals', '8. RG tool use'), 58: ('Limit decreases', '8. RG tool use'), 59: ('Time-outs / cool-offs', '8. RG tool use'), 60: ('Self-exclusions', '8. RG tool use'), 61: ('Time to self-exclusion', '8. RG tool use'), 62: ('Reality check dismissals', '8. RG tool use'), 63: ('RG page / self-test use', '8. RG tool use'), 64: ('National self-exclusion match', '8. RG tool use'), 65: ('Number of payment methods', '9. Payments'), 66: ('New payment methods added', '9. Payments'), 67: ('Registered cards', '9. Payments'), 68: ('Payment method type', '9. Payments'), 69: ('Third-party payment method', '9. Payments'), 70: ('Bonus usage rate', '10. Bonuses & marketing'), 71: ('Bonus begging', '10. Bonuses & marketing'), 72: ('Response to promotions', '10. Bonuses & marketing'), 73: ('Reward / free-spin chasing', '10. Bonuses & marketing'), 74: ('Customer service contacts', '11. Customer interaction'), 75: ('Complaints', '11. Customer interaction'), 76: ('Harm language in chat / email', '11. Customer interaction'), 77: ('Third-party contacts', '11. Customer interaction'), 78: ('RG interaction outcomes', '11. Customer interaction'), 79: ('Self-reported screening', '11. Customer interaction'), 80: ('Login frequency', '12. Account & device'), 81: ('Logins without betting', '12. Account & device'), 82: ('Device / channel', '12. Account & device'), 83: ('Account detail changes', '12. Account & device'), 84: ('Multiple brand accounts', '12. Account & device'), 85: ('Change vs own baseline', '13. Change & trend'), 86: ('Escalation slope', '13. Change & trend'), 87: ('Early behaviour window', '13. Change & trend'), 88: ('Look-back window', '13. Change & trend'), 89: ('Behaviour volatility', '13. Change & trend')}

COLS = [f"RG-{i:03d} {META[i][0]}" for i in range(1, 90)]

# Data points never used as model inputs
EXCLUDE = {
    79: "PGSI score - too close to the outcome (leakage)",
    78: "RG interaction outcome - the operator's own past action (circular)",
}

# Excel M1 scorecard: (RG id, rule, threshold or ('q', quantile of training data), weight)
M1_SCORECARD = [
    (20, ">=", ("q", .8), 3), (45, ">=", ("q", .8), 3), (60, ">=", 1, 3), (37, ">=", ("q", .8), 2),
    (33, ">=", ("q", .8), 2), (1, "<=", 25, 1), (12, ">=", ("q", .8), 2), (9, "<=", ("q", .2), 1),
    (21, ">=", 3, 2), (22, ">=", 2, 2), (30, ">=", 1, 2), (57, ">=", 1, 2), (49, ">=", 3, 1),
    (41, ">=", ("q", .8), 1),
]
M1_HIGH_CUTOFF = 40   # 'High' band in the Excel


def col(rg_id: int) -> str:
    """Column name for an RG id, e.g. 20 -> 'RG-020 Deposits per session'."""
    return COLS[rg_id - 1]


def rg_id(column: str) -> int:
    """RG id from a (possibly one-hot encoded) column name."""
    return int(column[3:6])
