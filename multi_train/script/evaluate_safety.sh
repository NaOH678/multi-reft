#!/bin/bash
set -euo pipefail

TASK_MODE=safety bash multi_train/script/evaluate_privacy.sh
