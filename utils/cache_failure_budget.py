"""A prefix is not representative when metadata is ordered by source group."""


def failure_budget_exceeded(failed, total_items, max_fraction):
    """Stop only once failures exceed this rank's complete assigned budget.

    The final global fraction remains separately checked before publication.
    Counts include restored failures; restarting must not reset the budget.
    """
    if failed < 0 or total_items < 0 or failed > total_items:
        raise ValueError('invalid cache failure counts')
    if not 0 <= max_fraction < 1:
        raise ValueError('invalid cache failure limit')
    return failed > total_items * max_fraction
