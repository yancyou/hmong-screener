"""
Joint task alias for UniSpeech-style baseline.

We reuse the exact same data pipeline and sample structure as
`gender_tone_repulsive_task.GenderToneRepulsiveTask` so we can swap criteria
without changing manifests or other training code.

Registered name: `gender_tone_unispeech`
"""

from fairseq.tasks import register_task

from fairseq.tasks.gender_tone_repulsive_task import (  # noqa: F401
    GenderToneRepulsiveTask,
    GenderToneRepulsiveTaskConfig,
)


@register_task("gender_tone_unispeech", dataclass=GenderToneRepulsiveTaskConfig)
class GenderToneUniSpeechTask(GenderToneRepulsiveTask):
    """Alias of `gender_tone_repulsive` task for UniSpeech baseline experiments."""

    pass

