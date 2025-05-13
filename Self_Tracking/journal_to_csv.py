import sys, csv, math, re
from docx import Document
import pandas as pd

CHUNK_SZ = 10_000  # pyxlsx has 32k limit on cell
# and we try to equalize chunk sizes but this has high variance
EXCLUDED = {"Plan", "Weekly Summaries, starting on Monday inclusive of Sunday."}
DATE_RE = re.compile(r"^\d{1,2}/\d{1,2}")

doc = Document(sys.argv[1])
rows = []
y = None
sec = None
buf = []


def emit_chunks(y, sec, buf):
    full = "\n".join(buf)
    if len(full) <= CHUNK_SZ:
        # fits in one cell → no suffix
        rows.append({"Year": y, "Section": sec, "Content": full})
        return

    L = len(full)
    n = math.ceil(L / CHUNK_SZ) or 1

    # split into day blocks
    days, dg = [], []
    for line in buf:
        if DATE_RE.match(line):
            if dg:
                days.append(dg)
            dg = [line]
        else:
            dg.append(line)
    if dg:
        days.append(dg)
    rem_chars = sum(len("\n".join(d)) for d in days)
    prev_chars_over = 0
    rem_chunks = n
    for i in range(1, n + 1):
        target = math.ceil(rem_chars / rem_chunks) - (prev_chars_over if rem_chunks > 1 else 0)
        chunk, c_chars = [], 0
        while days and c_chars + len("\n".join(days[0])) <= target:
            g = days.pop(0)
            chunk.extend(g)
            c_chars += len("\n".join(g))
        if not chunk and days:
            g = days.pop(0)
            chunk.extend(g)
            c_chars += len("\n".join(g))
        rows.append({"Year": y, "Section": f"{sec} ({i})", "Content": "\n".join(chunk)})
        prev_chars_over = target - c_chars
        rem_chars -= c_chars
        rem_chunks -= 1


for p in doc.paragraphs:
    style, t = p.style.name, p.text.rstrip()
    if not t:
        continue
    if style == "Heading 2":
        y = t
    elif style == "Heading 4":
        if sec and sec not in EXCLUDED:
            emit_chunks(y, sec, buf)
        sec = t
        buf = []
    else:
        if sec and sec not in EXCLUDED:
            buf.append(t)

if sec and sec not in EXCLUDED:
    emit_chunks(y, sec, buf)

df = pd.DataFrame(rows, columns=["Year", "Section", "Content"])
df.to_excel(sys.argv[2], engine="openpyxl", index=False)
