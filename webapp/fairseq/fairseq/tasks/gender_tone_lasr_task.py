"""
Joint task for LASR baseline.

This task intentionally reuses the exact same data pipeline and sample structure
as `gender_tone_repulsive_task.GenderToneRepulsiveTask` so we can swap criteria
without changing manifests or other training code.

Registered name: `gender_tone_lasr`
"""

from fairseq.tasks import register_task

from fairseq.tasks.gender_tone_repulsive_task import (  # noqa: F401
    GenderToneRepulsiveTask,
    GenderToneRepulsiveTaskConfig,
)


@register_task("gender_tone_lasr", dataclass=GenderToneRepulsiveTaskConfig)
class GenderToneLasrTask(GenderToneRepulsiveTask):
    """Alias of `gender_tone_repulsive` task for LASR experiments."""

    pass

