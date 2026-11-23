# CANNJudge template check

Run this check before submitting a candidate produced by an Ascend C workflow.
It compares the downloaded template with the candidate using the editable
flags from the CANNJudge CLI template manifest.

Download the original template with the
[CANNJudge CLI](https://github.com/spdc-elm/cannjudge-cli), and use a separate
copy for development. Keep the JSON manifest alongside the original template.

```bash
cannjudge template "$PROBLEM_URL" --out ./template --json > template.json
python3 check_template.py --manifest template.json --template template --candidate candidate
```

The checker requires Python 3.9 or newer and uses the standard library.
Its JSON output lists changed editable and read-only files declared in the
manifest. Exit codes are 0 for a valid comparison without read-only changes,
1 for read-only changes, and 2 for invalid inputs such as missing files.

Files outside the manifest are outside this check. Use the CANNJudge CLI
submission dry run to review the actual files selected for upload.
