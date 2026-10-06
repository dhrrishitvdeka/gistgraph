# Security policy

## Supported versions

Only the latest commit on `main` is supported. The project is pre-1.0 research code.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's private vulnerability
reporting (the **Security** tab of the repository, then **Report a vulnerability**) or email
dhrrishitvdeka@duck.com. Include what you found, how to reproduce it, and the affected files.

You can expect an acknowledgement within a week.

## Scope notes

- The code downloads datasets and model weights from the Hugging Face Hub and, for LongBench, reads
  a zip archive. Loading checkpoints with `torch.load` can execute code if the file is untrusted, so
  only load `compressor.pt` files you produced or trust.
- Experiment configs are plain YAML parsed with `yaml.safe_load`.
