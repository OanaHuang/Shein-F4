"""Preparation-phase guard. Real-data fitting/scoring is not authorized yet."""


class TrainingNotAuthorizedError(RuntimeError):
    pass


def require_preparation_mode(dry_run):
    if not dry_run:
        raise TrainingNotAuthorizedError(
            'Formal F4 training is not authorized. Only --dry-run and synthetic smoke tests are allowed.')


def require_synthetic(frame):
    if len(frame) > 200 or 'pkg_no' not in frame:
        raise TrainingNotAuthorizedError('Smoke data must have at most 200 synthetic rows')
    if not frame['pkg_no'].astype('string').str.startswith('SMOKE_').all():
        raise TrainingNotAuthorizedError('Synthetic smoke keys must start with SMOKE_')
