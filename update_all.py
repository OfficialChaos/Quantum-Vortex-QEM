"""
update_all.py
─────────────────────────────────────────────────────────────────
Run this after editing project_config.yaml to sync all local files.

Usage:
    python update_all.py

Updates:
    - paper/main.tex         (ZENODOID, GITHUBURL, AUTHORNAME, run counts)
    - paper/zenodo_metadata.yaml
    - paper/references.bib   (if self-citation present)
    - README.md              (DOI badge, citation block)
    - CITATION.cff           (DOI, version)
    - paper/figures captions (via main.tex only)

After running, manually update:
    - ORCID work entry
    - Academia.edu paper DOI
    - LinkedIn Zenodo link
    - GitHub profile README DOI badge
    - Zenodo record metadata
    - Overleaf (upload main.tex, recompile, download PDF)
"""

import yaml
import re
import os

# ── Load config ───────────────────────────────────────────────────
with open("project_config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

author      = cfg["author"]
project     = cfg["project"]
zenodo      = cfg["zenodo"]
arxiv       = cfg["arxiv"]
experiments = cfg["experiments"]

doi         = zenodo["doi"]
github_url  = project["github_url"]
title       = project["title"]
version     = project["version"]
year        = project["year"]
name        = author["name"]
orcid       = author["orcid"]
total_runs  = experiments["total_runs"]
arxiv_id    = arxiv["id"]
arxiv_url   = arxiv["url"]

print("=" * 60)
print("update_all.py — syncing all local files from project_config.yaml")
print(f"  DOI     : {doi}")
print(f"  Version : {version}")
print(f"  arXiv   : {arxiv_id}")
print(f"  Runs    : {total_runs}")
print("=" * 60)

# ── Helper ────────────────────────────────────────────────────────
def update_file(path, replacements):
    if not os.path.exists(path):
        print(f"  SKIP (not found): {path}")
        return
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    original = content
    for old, new in replacements:
        content = content.replace(old, new)
    if content != original:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  UPDATED: {path}")
    else:
        print(f"  NO CHANGE: {path}")

# ── main.tex ─────────────────────────────────────────────────────
update_file("paper/main.tex", [
    # Author
    (r'\newcommand{\AUTHORNAME}{' + name + r'}',
     r'\newcommand{\AUTHORNAME}{' + name + r'}'),
    # GitHub URL
    (r'\newcommand{\GITHUBURL}{' + github_url + r'}',
     r'\newcommand{\GITHUBURL}{' + github_url + r'}'),
    # Zenodo DOI — catch any previous DOI
    *[(r'\newcommand{\ZENODOID}{' + old_doi + r'}',
       r'\newcommand{\ZENODOID}{' + doi + r'}')
      for old_doi in [
          "10.5281/zenodo.18827720",
          "10.5281/zenodo.18827721",
          "10.5281/zenodo.18830306",
          doi
      ]],
    # arXiv badge in abstract
    ("arXiv:PENDING", f"arXiv:{arxiv_id}" if arxiv_id != "PENDING" else "arXiv:PENDING"),
])

# ── README.md ────────────────────────────────────────────────────
update_file("README.md", [
    # DOI badge — catch any previous DOI
    *[(f"https://doi.org/10.5281/zenodo.{old}",
       f"https://doi.org/{doi}")
      for old in ["18827720", "18827721", "18830306", doi.split("zenodo.")[1]]],
    # Citation DOI
    *[(f"doi    = {{10.5281/zenodo.{old}}},",
       f"doi    = {{{doi}}},")
      for old in ["18827720", "18827721", "18830306", doi.split("zenodo.")[1]]],
    # arXiv badge
    ("arXiv-pending-red.svg)]()",
     f"arXiv-pending-red.svg)]()" if arxiv_id == "PENDING"
     else f"arXiv-{arxiv_id}-red.svg)]({arxiv_url})"),
    # Checklist
    ("- [ ] arXiv submission (quant-ph)",
     f"- [x] arXiv submission — {arxiv_id}" if arxiv_id != "PENDING"
     else "- [ ] arXiv submission (quant-ph)"),
])

# ── CITATION.cff ─────────────────────────────────────────────────
update_file("CITATION.cff", [
    *[(f"doi: 10.5281/zenodo.{old}",
       f"doi: {doi}")
      for old in ["18827720", "18827721", "18830306", doi.split("zenodo.")[1]]],
    *[(f"version: v{old}",
       f"version: v{version}")
      for old in ["0.1.0", "0.2.0", "0.3.0", version]],
])

# ── zenodo_metadata.yaml ──────────────────────────────────────────
update_file("paper/zenodo_metadata.yaml", [
    *[(f"10.5281/zenodo.{old}",
       doi)
      for old in ["18827720", "18827721", "18830306", doi.split("zenodo.")[1]]],
    ("version: \"0.1.0\"", f"version: \"{version}\""),
    ("version: \"0.2.0\"", f"version: \"{version}\""),
    ("version: \"0.3.0\"", f"version: \"{version}\""),
    ("identifier: \"arXiv:PENDING\"",
     f"identifier: \"arXiv:{arxiv_id}\"" if arxiv_id != "PENDING"
     else "identifier: \"arXiv:PENDING\""),
])

print("")
print("─" * 60)
print("LOCAL FILES UPDATED. Now manually update:")
print("")
print("  [ ] ORCID        — https://orcid.org/0009-0002-2480-2430")
print(f"      Work DOI: {doi}")
print("")
print("  [ ] Academia.edu — update paper DOI field")
print(f"      DOI: {doi}")
print("")
print("  [ ] LinkedIn     — update Zenodo link in position")
print(f"      URL: https://doi.org/{doi}")
print("")
print("  [ ] GitHub profile README — update DOI badge")
print(f"      https://doi.org/{doi}")
print("")
print("  [ ] Zenodo record — verify title, author, ORCID")
print(f"      https://doi.org/{doi}")
print("")
if arxiv_id != "PENDING":
    print(f"  [ ] arXiv live — update all profiles with {arxiv_id}")
print("  [ ] Overleaf — upload main.tex, recompile, download PDF")
print("─" * 60)
print("")
print("Then commit:")
print("  git add .")
print(f"  git commit -m \"release: v{version} — DOI {doi}\"")
print("  git push origin main")
if arxiv_id != "PENDING":
    print(f"  git tag v{version}")
    print(f"  git push origin v{version}")
