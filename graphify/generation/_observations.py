"""Private delivery of lifecycle observations to one caller's adapter.

An observation adapter exists to render progress, so it has no authority over
the operation it is watching: it cannot change what is published, and a bug in
it cannot become a failed Graph generation. This module is the one place that
holds that rule, so no operation has to remember it at each emit site.
"""

from __future__ import annotations

import warnings

from graphify.generation._types import Observation, ObservationAdapter


class _Observations:
    """Deliver observations to an optional adapter, absorbing its failures."""

    def __init__(self, adapter: ObservationAdapter | None = None) -> None:
        """Bind one operation's reporting to the adapter that asked for it."""
        self._adapter = adapter
        self._dropped = False

    def emit(self, observation: Observation) -> None:
        """Report one lifecycle fact, or do nothing when nobody is watching.

        A raising adapter is warned about once and then dropped for the rest of
        the operation. Dropping rather than retrying is deliberate: an adapter
        that failed on one observation will almost always fail on the next, and
        a warning per emit would bury the operation's own output — which is the
        thing the adapter was supposed to be making readable.
        """
        if self._adapter is None or self._dropped:
            return
        try:
            self._adapter.observe(observation)
        except Exception as exc:  # noqa: BLE001 — an adapter may fail any way it likes
            self._dropped = True
            warnings.warn(
                "the observation adapter failed and was dropped for the rest of "
                f"this operation: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
