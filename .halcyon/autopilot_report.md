# Halcyon Autopilot Report
- Generated: 2025-10-17T20:48:24.314360+00:00
- Mode: full
- Branch: halcyon-improvements-20251017
- Adapter session: fallback
- Total Duration: 3.36s

## Task Outcomes
- `tests` (testing) → **SKIPPED** in 1.36s
  - Notes: No tests generated. Errors: Capsule generation failed for dataset_functions.py: Command '['/home/evo-nirvana/dev/projects/repos/pcbvision_halcyon/venv/bin/python3', 'tools/autopilot_codegen/main.py', 'tests']' returned non-zero exit status 2.; Fallback tests failed validation for dataset_functions.py (pytest may not be installed)
- `docs-docstrings` (documentation) → **SKIPPED** in 0.39s
  - Notes: No docstring gaps detected.
- `lint` (linting) → **FAILED** in 0.43s
  - Notes: compileall reported errors (exit code 1).
- `tests-verify` (testing) → **FAILED** in 0.40s
  - Notes: Task failed: dispatch unavailable (Halcyon dispatch for task tests-verify failed with exit code 1.
stdout:

stderr:
╭──────────────────); no local handler for RUN_TESTS
- `security` (security) → **SKIPPED** in 0.39s
  - Notes: Bandit not installed; skipped.; pip-audit not installed; skipped.
- `docs-readme` (documentation) → **SKIPPED** in 0.39s
  - Notes: README already contained autopilot section.

## Metrics
### Before
| Metric | Value |
| --- | --- |
| total_python_files | 9 |
| functions_without_docstrings | 0 |
| functions_without_type_hints | 11 |
| modules_without_tests | 9 |
| bare_except_blocks | 0 |
| logging_statements | 0 |
| todo_count | 55 |
| workflows_present | True |
| pre_commit_present | True |
| contributing_present | True |
| license_present | True |
| code_of_conduct_present | True |
| issue_templates_present | True |
| pr_template_present | True |
| changelog_present | True |
| docs_dir_present | False |
| coverage_files | 0 |
| requirements_files | 0 |

### After
| Metric | Value |
| --- | --- |
| total_python_files | 9 |
| functions_without_docstrings | 0 |
| functions_without_type_hints | 11 |
| modules_without_tests | 9 |
| bare_except_blocks | 0 |
| logging_statements | 0 |
| todo_count | 55 |
| workflows_present | True |
| pre_commit_present | True |
| contributing_present | True |
| license_present | True |
| code_of_conduct_present | True |
| issue_templates_present | True |
| pr_template_present | True |
| changelog_present | True |
| docs_dir_present | False |
| coverage_files | 0 |
| requirements_files | 0 |

## Recommendations
- Investigate `tests` (status=skipped). No tests generated. Errors: Capsule generation failed for dataset_functions.py: Command '['/home/evo-nirvana/dev/projects/repos/pcbvision_halcyon/venv/bin/python3', 'tools/autopilot_codegen/main.py', 'tests']' returned non-zero exit status 2.; Fallback tests failed validation for dataset_functions.py (pytest may not be installed)
- Investigate `docs-docstrings` (status=skipped). No docstring gaps detected.
- Investigate `lint` (status=failed). compileall reported errors (exit code 1).
- Investigate `tests-verify` (status=failed). Task failed: dispatch unavailable (Halcyon dispatch for task tests-verify failed with exit code 1.
stdout:

stderr:
╭──────────────────); no local handler for RUN_TESTS
- Investigate `security` (status=skipped). Bandit not installed; skipped.; pip-audit not installed; skipped.
- Investigate `docs-readme` (status=skipped). README already contained autopilot section.
