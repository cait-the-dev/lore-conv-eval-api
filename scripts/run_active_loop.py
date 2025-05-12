"""
cron weekly:
1. re-merge labels
2. train quick model & sample QA batch
   (manual QA happens in Label Studio)
3. after QA exported to `belief_labeled.csv`, the fine-tune job
   in `train_scripts/` retrains RoBERTa and bumps model version.
"""

from labeling_pipeline.merge import run as merge
from labeling_pipeline.active_learning import run as al

merge()
al()
