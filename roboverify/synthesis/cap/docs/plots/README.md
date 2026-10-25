# CaP report figures

Standalone plots generated from saved experiment artifacts are stored here and
embedded in their corresponding reports. Revision 4.2 figures are produced by
`synthesis.cap.revision42_report`; their numerical sources and audit live in the
analysis directory linked from [the report](../revision_4_2_report.md). Color
identifies the measure and hatching identifies the model.

Revision 4.3 prompt-factorial figures (pooled, plus `_baseline`, `_method1` and `_method2` per method) are produced by `synthesis.cap.revision43_prompt_report` from the main batch and its rerun ([report](../revision_4_3_report.md#16-cell-table-round-0-and-final-success-150-trials-per-cell)).

Revision 4.3 feedback-ablation figures (`revision43_feedback_effects`, `_rounds`, `_groups` and `_cost`, each pooled and with a `_by_method` version) are produced by `synthesis.cap.revision43_feedback_report` from the Stage B batch ([report](../revision_4_3_report.md#25-results)).

Do not hand-edit figure values or count pending trials as observed failures.
