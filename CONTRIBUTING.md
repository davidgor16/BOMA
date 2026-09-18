# Contributing

Please report the release or commit, model, experiment condition, package versions, full command, and relevant log excerpt when reporting a reproducibility problem. Include the artifact checksum when a downloaded file appears damaged. Do not upload private credentials or unrelated local files.

Keep archival results immutable. Run experiments into a new output directory. Propose algorithm changes separately from documentation and packaging changes, with a clear account of their effect on the experimental protocol.

For annotation-only changes, run `python tools/verify_source.py` against the release manifest. A deliberate algorithm change requires an explicit versioned update to that manifest and a new validation record.
