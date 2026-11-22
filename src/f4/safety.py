"""Preparation-mode guard; scoped real training uses the explicit backtest route."""


class TrainingNotAuthorizedError(RuntimeError):
    pass


def require_preparation_mode(dry_run):
    if not dry_run:
        raise TrainingNotAuthorizedError(
            'Preparation requires --dry-run; real fitting requires the explicit authorized training route.')


def require_synthetic(frame):
    if len(frame) > 200 or 'pkg_no' not in frame:
        raise TrainingNotAuthorizedError('Smoke data must have at most 200 synthetic rows')
    if not frame['pkg_no'].astype('string').str.startswith('SMOKE_').all():
        raise TrainingNotAuthorizedError('Synthetic smoke keys must start with SMOKE_')
