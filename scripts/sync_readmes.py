"""Generate localized READMEs from English layout, shared data and translation catalogs.

Run normally to regenerate; --check reports stale output without writing it.
All command blocks, numeric table cells and media URLs are copied from README.md.
Missing prose translations are errors rather than silently keeping an old paragraph.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = {
    "zh-CN": "简体中文", "zh-TW": "繁體中文", "ko": "한국어", "es": "Español",
    "ja": "日本語", "hi": "हिन्दी", "mr": "मराठी", "ru": "Русский",
    "pt": "Português", "te": "తెలుగు", "fr": "Français", "de": "Deutsch",
    "it": "Italiano", "vi": "Tiếng Việt", "ar": "العربية", "ur": "اردو",
    "bn": "বাংলা", "fa": "فارسی", "ro": "Română", "tr": "Türkçe",
}
RTL = {"ar", "ur", "fa"}


def blocks(text: str) -> list[str]:
    lines = text.splitlines()
    result = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        start = i
        line = lines[i].strip()
        if line.startswith("```"):
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                i += 1
            i += 1
        elif line == "<!-- BEGIN QUALITATIVE GALLERY -->":
            while i < len(lines) and lines[i].strip() != "<!-- END QUALITATIVE GALLERY -->":
                i += 1
            i += 1
        elif line == "<table>":
            while i < len(lines) and lines[i].strip() != "</table>":
                i += 1
            i += 1
        elif line.startswith("|"):
            while i < len(lines) and lines[i].strip().startswith("|"):
                i += 1
        else:
            i += 1
            while i < len(lines) and lines[i].strip():
                i += 1
        result.append("\n".join(lines[start:i]))
    return result


def normalized(text: str) -> str:
    return " ".join(text.split())


def message_id(text: str) -> str:
    return hashlib.sha256(normalized(text).encode()).hexdigest()[:16]


def literal(text: str) -> bool:
    # Technical names, numbers, symbols and executable options stay identical.
    return (not text.strip() or not re.search(r"[A-Za-z]", text)
            or bool(re.fullmatch(r"<code>[^<]*</code>|`[^`]+`", text))
            or text in {"BF16", "INT8 (W8A8)", "TP", "SP", "Full", "VDN · official",
                        "VSA V2 · 80%", "Veda · 90%", "Spark-H3 · 90%"}
            or bool(re.fullmatch(r"[\d .ms×%†TL()<br>/—-]+", text)))


def messages(source: str) -> dict[str, str]:
    found = {}
    def add(value):
        if not literal(value):
            found[message_id(value)] = normalized(value)
    for block in blocks(source):
        if block.startswith("```") or block.startswith(("<p ", "<h1", "<div", "</div", "<a ")):
            continue
        if "**English** |" in block or block.startswith(("[![", "> **License:")):
            continue
        if block.startswith("<!--") and "QUALITATIVE GALLERY" not in block:
            continue
        if block.startswith("|"):
            for line in block.splitlines():
                if re.match(r"\|\s*[-:]", line):
                    continue
                for cell in line.strip("|").split("|"):
                    add(cell.strip())
        elif "<table>" in block:
            for value in re.findall(r"<t[hd]\b[^>]*>(.*?)</t[hd]>", block, re.S):
                # Gallery cells contain Markdown captions and unchanged video URLs.
                if "github.com/user-attachments/" in value:
                    for caption in re.findall(r"\*\*.*?\*\*", value):
                        add(caption)
                elif not re.search(r"<td|<th", value):
                    add(value.strip())
        elif block.startswith(("![", "[")):
            for label in re.findall(r"\[([^\]]+)\]\([^)]+\)", block):
                add(label)
        else:
            add(block)
    return found


def relativize(text: str, prefix: str) -> str:
    def path(target):
        if not prefix or re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:|#|/", target):
            return target
        return prefix + target
    text = re.sub(r"(\]\()([^\s)]+)(\))", lambda m: m[1] + path(m[2]) + m[3], text)
    return re.sub(r'(\b(?:href|src)=")([^"]+)(")', lambda m: m[1] + path(m[2]) + m[3], text)


def render(source: str, locale: str, catalog: dict) -> str:
    required = messages(source)
    translations = catalog["messages"]
    missing = set(required) - set(translations)
    if missing:
        raise ValueError(f"{locale}: missing translations: " + "; ".join(
            f"{key}: {required[key]}" for key in sorted(missing)))
    def translated(value):
        if literal(value):
            return value
        return translations[message_id(value)]
    result = []
    for block in blocks(source):
        if block.startswith("```"):
            result.append(('<div dir="ltr">\n\n' + block + '\n\n</div>') if locale in RTL else block)
            continue
        if '<a href="#install">' in block:
            for original, label in zip(("Installation", "Models", "Serve", "Performance"), catalog["navigation"]):
                block = block.replace(">" + original + "</a>", ">" + html.escape(label) + "</a>")
        elif "**English** |" in block:
            block = block.replace("**English**", "[English](README.md)")
            target = "README.zh-CN.md" if locale == "zh-CN" else f"i18n/README.{locale}.md"
            block = block.replace(f"[{LANGUAGES[locale]}]({target})", f"**{LANGUAGES[locale]}**")
        elif block.startswith("|"):
            lines = []
            for line in block.splitlines():
                if re.match(r"\|\s*[-:]", line):
                    lines.append(line)
                else:
                    lines.append("| " + " | ".join(translated(c.strip()) for c in line.strip("|").split("|")) + " |")
            block = "\n".join(lines)
        elif "<table>" in block:
            def cell(match):
                value = match[2]
                if "github.com/user-attachments/" in value:
                    value = re.sub(r"\*\*.*?\*\*", lambda m: translated(m[0]), value)
                elif not re.search(r"<td|<th", value):
                    value = translated(value.strip())
                return match[1] + value + match[3]
            block = re.sub(r"(<t[hd]\b[^>]*>)(.*?)(</t[hd]>)", cell, block, flags=re.S)
        elif block.startswith(("![", "[")) and not block.startswith("[!["):
            block = re.sub(r"\[([^\]]+)\](\([^)]+\))", lambda m: "[" + translated(m[1]) + "]" + m[2], block)
        elif not block.startswith(("<", "[![", "> **License:")):
            block = translated(block)
        block = relativize(block, "" if locale == "zh-CN" else "../")
        if locale in RTL and ("<table>" in block or block.startswith("|")):
            block = '<div dir="ltr">\n\n' + block + '\n\n</div>'
        result.append(block)
    digest = hashlib.sha256(source.encode()).hexdigest()
    header = f"<!-- Generated by scripts/sync_readmes.py; AI-assisted translation; source-sha256: {digest} -->\n"
    body = "\n\n".join(result)
    if locale in RTL:
        body = '<div dir="rtl">\n\n' + body + '\n\n</div>'
    return header + body + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--extract", action="store_true", help="List required English message IDs")
    args = parser.parse_args()
    source = (ROOT / "README.md").read_text()
    if args.extract:
        print(json.dumps(messages(source), indent=2, ensure_ascii=False))
        return
    reference = ROOT / "i18n/catalogs/source.json"
    if json.loads(reference.read_text()) != messages(source):
        sys.exit("English messages changed. Refresh i18n/catalogs/source.json with --extract and update translations.")
    stale = []
    for locale in LANGUAGES:
        catalog_path = ROOT / f"i18n/catalogs/{locale}.json"
        expected = render(source, locale, json.loads(catalog_path.read_text()))
        target = ROOT / ("README.zh-CN.md" if locale == "zh-CN" else f"i18n/README.{locale}.md")
        if args.check:
            if not target.exists() or target.read_text() != expected:
                stale.append(str(target.relative_to(ROOT)))
        else:
            target.write_text(expected)
    if stale:
        sys.exit("Out-of-sync READMEs: " + ", ".join(stale) + ". Run python scripts/sync_readmes.py")
    print(f"{'Checked' if args.check else 'Updated'} {len(LANGUAGES)} localized READMEs")


if __name__ == "__main__":
    main()
