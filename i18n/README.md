# Keeping the READMEs synchronized

`README.md` is the English source. `scripts/sync_readmes.py` generates the
Simplified Chinese README and the 19 other language versions using the same
section order. Commands, model paths, numeric table cells, benchmark records,
MP4 URLs and figure paths come directly from that source. Navigation labels
and explanatory prose come from `catalogs/<language>.json`.

Translations are AI-assisted and should be reviewed by speakers of the target
language. The English version remains the technical reference. The generator
fails on missing translations rather than retaining stale prose or quietly
substituting English.

After editing the English README:

```bash
# Refresh the English message list if prose or table headings changed.
python3 scripts/sync_readmes.py --extract > i18n/catalogs/source.json
# Add/update the corresponding message IDs in each language catalog.
python3 scripts/sync_readmes.py
python3 scripts/sync_readmes.py --check
```

Code, media and numeric-only changes propagate when regenerating; they do not
need separate translations. Message IDs are hashes of whitespace-normalized
English text, so formatting-only reflow does not discard a translation.
Do not edit generated READMEs directly. Fix the relevant catalog and regenerate.
Right-to-left languages keep commands and data tables explicitly left-to-right.
The supplied `.githooks/pre-commit` checks the staged snapshot when the source,
catalogs, generator or translated READMEs change. Enable it for this checkout:

```bash
git config --local core.hooksPath .githooks
```

It checks synchronization without NPU/GPU access, network APIs or translation keys.
`python3 scripts/sync_readmes.py --check` can also be used by CI. GitHub Actions
installation requires a credential with the `workflow` scope; no workflow is
installed by this update.

## 中文维护说明

英文 `README.md` 为内容源，简体中文及其余 19 个语言版本统一生成。命令、数值、
性能表、视频和图片共用源内容；正文和导航在各语言 JSON 中维护，链接按目录自动调整。
更新英文后，刷新消息列表、补齐变动段落的翻译，再运行上面的生成与检查命令。
启用仓库的提交前检查后，缺失翻译或已暂存的生成文件过期时会报错，
不会悄悄保留旧段落。检查基于实际暂存内容，避免仅更新工作区却漏提交翻译。
