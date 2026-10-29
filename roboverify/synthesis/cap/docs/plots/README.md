# CaP report figures

Standalone plots generated from saved experiment artifacts are stored here and
embedded in their corresponding reports. Revision 4.2 figures are produced by
`synthesis.cap.reports.revision42_report`; their numerical sources and audit live in the
analysis directory linked from [the report](../revision_4_2_report.md). Color
identifies the measure and hatching identifies the model.

Revision 4.3 prompt-factorial figures (pooled, plus `_baseline`, `_method1` and `_method2` per method) are produced by `synthesis.cap.reports.revision43_prompt_report` from the main batch and its rerun ([report](../revision_4_3_report.md#16-cell-table-round-0-and-final-success-150-trials-per-cell)).

Revision 4.3 feedback-ablation figures (`revision43_feedback_effects`, `_rounds`, `_groups` and `_cost`, each pooled and with a `_by_method` version) are produced by `synthesis.cap.reports.revision43_feedback_report` from the Stage B batch ([report](../revision_4_3_report.md#25-results)).

Final-experiment figures (`final_v1_success`, `_recovery`, `_round0_outcomes`, `_cost` and `_cache`) are produced by `synthesis.cap.reports.final_report` from `runs/cap-final-v2/20261027-final-50` and its supplement ([report](../final_v1_report.md)).

Demonstrations-off figures (`final_v2_wo_demo_success`, `_recovery`, `_round0_outcomes`, `_cost`, `_cache` and `_partial_reward`, the last with the demonstrations-on run as a comparison arm) are produced by the same module from `runs/cap-final-v2-wo-demo/20261028-final-50` ([report](../final_v2_wo_demo_report.md)).

Block-count generalization figures (`revision9_success`, `_partial_reward`, `_trial_heatmap`, `_largest_count`, `_outcomes` and `_outcomes_by_count`) are produced by `synthesis.cap.reports.generalization_report` from `runs/cap-generalization/20261029-full-eval` ([report](../revision_9_report.md)).

Do not hand-edit figure values or count pending trials as observed failures.
