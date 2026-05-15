# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import pathlib

from openpi.shared import normalize as _normalize


_BEHAVIOR_CHALLENGE_ASSET_ID = "physical-intelligence/behavior"
_BEHAVIOR_CHALLENGE_FALLBACK_DIR = pathlib.Path(
    "assets/behavior-1k/2025-challenge-demos"
)


def load_norm_stats(checkpoint_dir, asset_id, checkpoints_module):
    """Load OpenPI norm stats, with a fallback for B1K challenge exports."""
    try:
        return checkpoints_module.load_norm_stats(checkpoint_dir, asset_id)
    except FileNotFoundError as exc:
        if asset_id != _BEHAVIOR_CHALLENGE_ASSET_ID:
            raise

        fallback_dir = pathlib.Path(checkpoint_dir) / _BEHAVIOR_CHALLENGE_FALLBACK_DIR
        fallback_file = fallback_dir / "norm_stats.json"
        if not fallback_file.exists():
            raise exc

        logging.info("Loaded norm stats from fallback path %s", fallback_dir)
        return _normalize.load(fallback_dir)
