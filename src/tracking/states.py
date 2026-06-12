"""Canonical tracking status values shared across the pipeline.

The lifecycle of a target is:

``NO_TARGET`` -> ``TRACKING`` <-> ``PREDICTING`` -> ``REACQUIRING`` -> ``TARGET_STALE``

* ``TRACKING``    -- a confirmed observation from the tracker this frame.
* ``PREDICTING``  -- short loss; the Kalman prediction carries the box.
* ``REACQUIRING`` -- local/global search is actively trying to re-find it.
* ``TARGET_STALE``-- not confirmed for too long; the box must not be trusted.
"""

from __future__ import annotations

NO_TARGET = "NO TARGET"
TRACKING = "TRACKING"
PREDICTING = "PREDICTING"
REACQUIRING = "REACQUIRING"
TARGET_STALE = "TARGET STALE"

ALL_STATES = (NO_TARGET, TRACKING, PREDICTING, REACQUIRING, TARGET_STALE)
