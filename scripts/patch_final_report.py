"""Patch Rayanah Final Report Draft.docx to fix structural issues.

Fixes:
- Remove duplicate "Proposed Solution Overview" section in Chapter 1
- Remove preliminary Functional/Non-functional requirement subsections from Chapter 1
- Convert [Normal] paragraphs with "N.N" or "N.N.N" prefixes in Chapters 3-8 to Heading 2/3
- Add consistent numbered prefixes to all Heading 2 / Heading 3 items in Chapters 1 and 2
- Insert Acknowledgments section after Abstract
- Insert auto-updating Table of Contents after Acknowledgments
- Ensure Heading 1 style uses "Chapter N: Title" format (already present)

Run with:
    python scripts/patch_final_report.py
Produces: "Rayanah Final Report Draft (Updated).docx"
"""
from __future__ import annotations

import copy
import re
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# ─── Paths ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "Rayanah Final Report Draft.docx"
DST = ROOT / "Rayanah Final Report Draft (Updated).docx"


# ─── Helpers ────────────────────────────────────────────
NUM_H2 = re.compile(r"^(\d+)\.(\d+)\s+(.+)$")
NUM_H3 = re.compile(r"^(\d+)\.(\d+)\.(\d+)\s+(.+)$")


def set_paragraph_style(par, style_name: str) -> None:
    par.style = par.part.document.styles[style_name]


def delete_paragraph(par) -> None:
    el = par._element
    el.getparent().remove(el)


def insert_paragraph_after(paragraph, text: str, style: str = "Normal"):
    """Insert a paragraph immediately after the given one, returning it."""
    new_p = OxmlElement("w:p")
    paragraph._element.addnext(new_p)
    from docx.text.paragraph import Paragraph
    new_par = Paragraph(new_p, paragraph._parent)
    new_par.style = paragraph.part.document.styles[style]
    if text:
        new_par.add_run(text)
    return new_par


def insert_toc_after(paragraph):
    """Insert a Word field that auto-generates a Table of Contents."""
    heading = insert_paragraph_after(paragraph, "Table of Contents", style="Heading 1")

    toc_par = insert_paragraph_after(heading, "", style="Normal")
    run = toc_par.add_run()

    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")

    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = r'TOC \o "1-3" \h \z \u'

    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")

    placeholder = OxmlElement("w:t")
    placeholder.text = "Right-click and select 'Update Field' to populate contents."

    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")

    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_sep)
    run._r.append(placeholder)
    run._r.append(fld_end)

    return toc_par


# ─── Main transformation ────────────────────────────────
def patch(src: Path, dst: Path) -> None:
    doc = Document(str(src))
    body = doc.element.body
    paragraphs = list(doc.paragraphs)

    # ── Pass 1: mark paragraphs for deletion / retitling ──
    to_delete_idx: set[int] = set()

    # Find key anchors
    abstract_idx = None
    ch1_start = None
    ch2_start = None
    ch3_start = None
    ch_end = len(paragraphs)

    for i, p in enumerate(paragraphs):
        text = p.text.strip()
        if text == "Abstract" and abstract_idx is None:
            abstract_idx = i
        if text.startswith("Chapter 1:"):
            ch1_start = i
        elif text.startswith("Chapter 2:"):
            ch2_start = i
        elif text.startswith("Chapter 3:"):
            ch3_start = i

    # Identify duplicate Proposed Solution Overview in Chapter 1
    # The legitimate one is "1.7 Proposed Solution Overview" (Heading 2, numbered)
    # The duplicate is a later unnumbered "Proposed Solution Overview" (Heading 2)
    solution_overview_indices = []
    if ch1_start is not None and ch2_start is not None:
        for i in range(ch1_start, ch2_start):
            p = paragraphs[i]
            if "Proposed Solution Overview" in p.text and p.style.name.startswith("Heading"):
                solution_overview_indices.append(i)

    # Keep the first (1.7), delete the second + its contents up until "Figure 1" caption or
    # next Heading 1 (Chapter 2).
    if len(solution_overview_indices) >= 2:
        dup_start = solution_overview_indices[1]
        # delete from dup_start until we hit either the Figure 1 caption or Chapter 2
        for j in range(dup_start, ch2_start):
            p = paragraphs[j]
            text = p.text.strip()
            # stop before Figure 1 caption (we keep that for the legitimate section)
            if text.startswith("Figure 1"):
                break
            to_delete_idx.add(j)

    # Remove preliminary Functional/Non-functional requirements subsections from Chapter 1
    # (they belong in Chapter 3 where the finalised versions already exist)
    if ch1_start is not None and ch2_start is not None:
        i = ch1_start
        while i < ch2_start:
            p = paragraphs[i]
            text = p.text.strip()
            if text in (
                "Functional requirements (preliminary)",
                "Non-functional requirements (preliminary)",
            ):
                # delete this heading and all subsequent list paragraphs until next heading
                j = i
                to_delete_idx.add(j)
                j += 1
                while j < ch2_start:
                    q = paragraphs[j]
                    if q.style.name.startswith("Heading"):
                        break
                    to_delete_idx.add(j)
                    j += 1
                i = j
            else:
                i += 1

    # ── Pass 2: promote [Normal] paragraphs beginning with N.N or N.N.N
    # to Heading 2 / Heading 3 across Chapters 3 onwards.
    if ch3_start is not None:
        for i in range(ch3_start, ch_end):
            if i in to_delete_idx:
                continue
            p = paragraphs[i]
            if p.style.name != "Normal":
                continue
            text = p.text.strip()
            if NUM_H3.match(text):
                set_paragraph_style(p, "Heading 3")
            elif NUM_H2.match(text):
                set_paragraph_style(p, "Heading 2")

    # ── Pass 3: apply numbered prefixes consistently to Chapter 1 and Chapter 2
    # Heading 2 entries so TOC numbering is uniform.
    def number_headings(start, end, chapter_num):
        h2_count = 0
        h3_count = 0
        for i in range(start, end):
            if i in to_delete_idx:
                continue
            p = paragraphs[i]
            style = p.style.name
            text = p.text.strip()
            if style == "Heading 2":
                # Skip if already numbered (e.g. "1.7 Proposed Solution Overview")
                if re.match(rf"^{chapter_num}\.\d+\s+", text):
                    # extract count to keep sequence
                    m = re.match(rf"^{chapter_num}\.(\d+)\s+", text)
                    h2_count = max(h2_count, int(m.group(1)))
                    h3_count = 0
                    continue
                h2_count += 1
                h3_count = 0
                p.text = f"{chapter_num}.{h2_count} {text}"
            elif style == "Heading 3":
                if re.match(rf"^{chapter_num}\.\d+\.\d+\s+", text):
                    continue
                h3_count += 1
                p.text = f"{chapter_num}.{h2_count}.{h3_count} {text}"

    if ch1_start is not None and ch2_start is not None:
        number_headings(ch1_start, ch2_start, 1)
    if ch2_start is not None and ch3_start is not None:
        number_headings(ch2_start, ch3_start, 2)

    # ── Pass 4: actually delete marked paragraphs (reverse order to preserve idx)
    for i in sorted(to_delete_idx, reverse=True):
        delete_paragraph(paragraphs[i])

    # Refresh paragraph list after deletions
    paragraphs = list(doc.paragraphs)

    # ── Pass 4b: sort References section alphabetically & drop duplicates
    ref_start = None
    ref_end = None
    for i, p in enumerate(paragraphs):
        if p.style.name == "Heading 1" and p.text.strip() == "References":
            ref_start = i
        elif ref_start is not None and p.style.name == "Heading 1":
            ref_end = i
            break
    if ref_start is not None:
        ref_end = ref_end or len(paragraphs)
        # Collect non-empty reference paragraphs
        ref_pars = []
        for j in range(ref_start + 1, ref_end):
            q = paragraphs[j]
            if q.text.strip():
                ref_pars.append(q)

        # Sort key: first author surname (text before first comma or first space-parenthesis)
        def sort_key(par):
            t = par.text.strip()
            m = re.match(r"^([A-Z][A-Za-z'\-]+)", t)
            return (m.group(1).lower() if m else t.lower(), t.lower())

        # Deduplicate by normalised prefix (author + year)
        seen = set()
        unique_pars = []
        for par in ref_pars:
            t = par.text.strip()
            m = re.match(r"^([A-Z][A-Za-z'\-\s.,]+?\(\d{4}[a-z]?\))", t)
            key = m.group(1).lower().replace(" ", "") if m else t.lower()[:40]
            if key in seen:
                delete_paragraph(par)
                continue
            seen.add(key)
            unique_pars.append(par)

        # Sort by moving the paragraph XML elements themselves; this preserves
        # hyperlinks and run formatting.
        sorted_pars = sorted(unique_pars, key=sort_key)
        if sorted_pars:
            anchor = sorted_pars[0]._element
            # Detach all sorted elements, then re-insert in new order after the
            # "References" heading element.
            for par in sorted_pars:
                el = par._element
                el.getparent().remove(el)
            ref_heading_el = paragraphs[ref_start]._element
            insert_after = ref_heading_el
            for par in sorted_pars:
                insert_after.addnext(par._element)
                insert_after = par._element

        paragraphs = list(doc.paragraphs)

    # ── Pass 5: add Acknowledgments section + TOC after Abstract
    abstract_idx = None
    for i, p in enumerate(paragraphs):
        if p.text.strip() == "Abstract":
            abstract_idx = i
            break

    # Find the last paragraph of the Abstract (the paragraph before the next
    # Heading 1, which is "Chapter 1: Introduction")
    if abstract_idx is not None:
        abstract_heading = paragraphs[abstract_idx]
        last_abstract_par = abstract_heading
        for j in range(abstract_idx + 1, len(paragraphs)):
            p = paragraphs[j]
            if p.style.name.startswith("Heading 1"):
                break
            last_abstract_par = p

        # Check whether Acknowledgments already exists; if yes, skip insertion
        has_ack = any(
            par.text.strip() == "Acknowledgments" and par.style.name.startswith("Heading")
            for par in paragraphs
        )
        if not has_ack:
            ack_body = insert_paragraph_after(
                last_abstract_par,
                (
                    "I would like to sincerely thank my project supervisor, "
                    "Piyajith Wijetunge, for his continued guidance, technical feedback, "
                    "and encouragement throughout the development of EcoWeatherFit. "
                    "I am also grateful to the School of Electronic Engineering and "
                    "Computer Science at Queen Mary University of London for providing "
                    "the academic environment and resources that made this project "
                    "possible. Finally, I would like to thank the participants of the "
                    "user acceptance testing sessions for their time and constructive "
                    "feedback."
                ),
                style="Normal",
            )
            ack_heading = insert_paragraph_after(
                last_abstract_par, "Acknowledgments", style="Heading 1"
            )
            # Note: insert_paragraph_after inserts directly after the anchor,
            # so the heading ends up between last_abstract_par and ack_body.
            # Re-fetch after insertion
            paragraphs = list(doc.paragraphs)
            # Find the acknowledgments body paragraph (last paragraph we just added)
            ack_body_par = None
            for par in paragraphs:
                if par.text.startswith("I would like to sincerely thank"):
                    ack_body_par = par
                    break

            # Insert the TOC after Acknowledgments body
            has_toc = any(
                par.text.strip().startswith("Table of Contents") for par in paragraphs
            )
            if ack_body_par is not None and not has_toc:
                insert_toc_after(ack_body_par)

    doc.save(str(dst))
    print(f"Wrote: {dst}")


if __name__ == "__main__":
    patch(SRC, DST)
