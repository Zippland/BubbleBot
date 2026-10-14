# Bubbles Skills

This directory contains built-in skills that extend Bubbles's capabilities.

Third-party `docx`, `pdf`, `pptx`, and `xlsx` skill bundles are not included in
this open-source distribution because their licenses restrict redistribution.
The skill extension interface remains available; add only skills you are
authorized to use and distribute. See the repository's `THIRD_PARTY_NOTICES.md`.

## Skill Format

Each skill is a directory containing a `SKILL.md` file with:
- YAML frontmatter (name, description, metadata)
- Markdown instructions for the agent

## Available Skills

| Skill | Description |
|-------|-------------|
| `github` | Interact with GitHub using the `gh` CLI |
| `weather` | Get weather info using wttr.in and Open-Meteo |
| `summarize` | Summarize URLs, files, and YouTube videos |
| `tmux` | Remote-control tmux sessions |
| `memory` | Manage agent memory |
| `skill-creator` | Create new skills |
