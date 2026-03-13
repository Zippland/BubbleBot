"""Skills loader for agent capabilities."""

import re
from pathlib import Path


class SkillsLoader:
    """
    Loader for agent skills.

    Skills are markdown files (SKILL.md) in session_dir/skills/{skill-name}/.
    The agent reads the full SKILL.md when it needs to use a skill.
    """

    def __init__(self, session_dir: Path):
        self.session_dir = session_dir
        self.skills_dir = session_dir / "skills"

    def list_skills(self) -> list[dict[str, str]]:
        """
        List all skills from session directory.

        Returns:
            List of skill info dicts with 'name', 'description'.
        """
        skills = []

        if self.skills_dir.exists():
            for skill_dir in self.skills_dir.iterdir():
                if skill_dir.is_dir():
                    skill_file = skill_dir / "SKILL.md"
                    if skill_file.exists():
                        desc = self._get_description(skill_file)
                        skills.append({"name": skill_dir.name, "description": desc})

        return skills

    def _get_description(self, skill_file: Path) -> str:
        """Extract description from SKILL.md frontmatter."""
        try:
            content = skill_file.read_text(encoding="utf-8")
            if content.startswith("---"):
                match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
                if match:
                    for line in match.group(1).split("\n"):
                        if line.startswith("description:"):
                            return line.split(":", 1)[1].strip().strip("\"'")
        except Exception:
            pass
        return ""

    def build_skills_summary(self) -> str:
        """
        Build a summary of all skills for the system prompt.

        Returns:
            XML-formatted skills list.
        """
        skills = self.list_skills()
        if not skills:
            return ""

        def escape_xml(s: str) -> str:
            return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        def escape_attr(s: str) -> str:
            return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

        lines = [
            "<skills>",
            "",
            "The following skills extend your capabilities. To use a skill, read its SKILL.md file using the read_file tool.",
            "",
        ]
        for s in skills:
            name = escape_attr(s["name"])
            desc = escape_xml(s["description"]) if s["description"] else name
            path = f"skills/{s['name']}/SKILL.md"
            lines.append(f'  <skill name="{name}" path="{path}">')
            lines.append(f"    {desc}")
            lines.append(f"  </skill>")
        lines.append("</skills>")

        return "\n".join(lines)

    def get_always_skills(self) -> list[str]:
        """Get skills marked as always=true in frontmatter."""
        result = []

        if not self.skills_dir.exists():
            return result

        for skill_dir in self.skills_dir.iterdir():
            if skill_dir.is_dir():
                skill_file = skill_dir / "SKILL.md"
                if skill_file.exists():
                    try:
                        content = skill_file.read_text(encoding="utf-8")
                        if content.startswith("---"):
                            match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
                            if match:
                                for line in match.group(1).split("\n"):
                                    if line.strip().startswith("always:"):
                                        val = line.split(":", 1)[1].strip().lower()
                                        if val == "true":
                                            result.append(skill_dir.name)
                                            break
                    except Exception:
                        pass

        return result

    def load_skills_for_context(self, skill_names: list[str]) -> str:
        """
        Load specific skills for inclusion in agent context.

        Args:
            skill_names: List of skill names to load.

        Returns:
            Formatted skills content.
        """
        parts = []
        for name in skill_names:
            skill_file = self.skills_dir / name / "SKILL.md"
            if skill_file.exists():
                content = skill_file.read_text(encoding="utf-8")
                # Strip frontmatter
                if content.startswith("---"):
                    match = re.match(r"^---\n.*?\n---\n", content, re.DOTALL)
                    if match:
                        content = content[match.end():].strip()
                parts.append(f"### Skill: {name}\n\n{content}")
        
        return "\n\n---\n\n".join(parts) if parts else ""
