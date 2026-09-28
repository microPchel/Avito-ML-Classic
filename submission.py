"""Checks for the competition submission format."""

import numpy as np


def validate_submission(submission, test_cookie_ids):
    assert submission.cookie_id.is_unique and len(submission) == len(test_cookie_ids)
    assert np.array_equal(submission.cookie_id, test_cookie_ids)
    assert submission.score.notna().all() and submission.score.between(0, 1).all()
