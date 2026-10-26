"""Reward that varies inside a GRPO group.

A constant score makes every advantage zero. Scoring from the response text
keeps at least some groups from tying, so the policy gradient is nonzero.
"""


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    del data_source, ground_truth, extra_info, kwargs
    text = solution_str.strip()
    if not text:
        return 0.0
    return float(sum(text.encode("utf-8")) % 5) / 4.0
