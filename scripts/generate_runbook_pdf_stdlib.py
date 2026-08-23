from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "runbook.md"
OUTPUT = ROOT / "docs" / "VibraGuard_Runbook.pdf"
PAGE_WIDTH = 595
PAGE_HEIGHT = 842
LEFT = 42
TOP = 800
LINE_HEIGHT = 14
MAX_CHARS = 92


def pdf_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def wrap_line(line: str, width: int = MAX_CHARS):
    if not line:
        return [""]
    result = []
    while len(line) > width:
        split_at = line.rfind(" ", 0, width + 1)
        if split_at < 1:
            split_at = width
        result.append(line[:split_at])
        line = line[split_at:].lstrip()
    result.append(line)
    return result


def document_lines():
    in_code = False
    for raw in SOURCE.read_text(encoding="utf-8").splitlines():
        if raw.startswith("```"):
            in_code = not in_code
            continue
        if raw.startswith("# "):
            yield ("title", raw[2:].strip())
        elif raw.startswith("## "):
            yield ("heading", raw[3:].strip())
        elif raw.startswith("### "):
            yield ("heading", raw[4:].strip())
        elif raw.startswith("- "):
            yield ("bullet", raw[2:].strip())
        elif raw.startswith("1. "):
            yield ("body", raw.strip())
        elif raw:
            yield ("code" if in_code else "body", raw)
        else:
            yield ("space", "")


def make_pages():
    pages = [[]]
    for kind, line in document_lines():
        width = 105 if kind == "code" else MAX_CHARS
        expanded = wrap_line(line, width) if kind != "space" else [""]
        for text in expanded:
            if len(pages[-1]) >= 50:
                pages.append([])
            pages[-1].append((kind, text))
        if kind in {"heading", "title"} and len(pages[-1]) >= 47:
            pages.append([])
    return [page for page in pages if page]


def content_stream(page):
    commands = ["BT"]
    y = TOP
    for kind, text in page:
        if kind == "title":
            font, size, color = "/F1", 18, "0.04 0.28 0.31 rg"
        elif kind == "heading":
            font, size, color = "/F1", 12, "0.04 0.35 0.38 rg"
        elif kind == "code":
            font, size, color = "/F2", 7.5, "0.08 0.20 0.22 rg"
        elif kind == "bullet":
            font, size, color = "/F3", 8.5, "0.10 0.22 0.24 rg"
            text = "- " + text
        else:
            font, size, color = "/F3", 8.5, "0.10 0.22 0.24 rg"
        if kind == "space":
            y -= 6
            continue
        commands.append(f"{color} {font} {size} Tf {LEFT} {y} Td ({pdf_escape(text)}) Tj")
        y -= LINE_HEIGHT if kind != "title" else 22
    commands.append("ET")
    return "\n".join(commands).encode("latin-1", errors="replace")


def build_pdf():
    pages = make_pages()
    objects = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    page_ids = []
    next_id = 4 + len(pages) * 2
    objects.append(b"")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
    for page in pages:
        content = content_stream(page)
        content_id = len(objects) + 1
        objects.append(b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream")
        page_id = len(objects) + 1
        objects.append((f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] /Resources << /Font << /F1 3 0 R /F2 4 0 R /F3 5 0 R >> >> /Contents {content_id} 0 R >>").encode())
        page_ids.append(page_id)
    objects[1] = ("<< /Type /Pages /Kids [" + " ".join(f"{pid} 0 R" for pid in page_ids) + f"] /Count {len(page_ids)} >>").encode()
    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode())
        output.extend(obj)
        output.extend(b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    OUTPUT.write_bytes(output)
    print(f"Created {OUTPUT} ({len(output)} bytes, {len(pages)} pages)")


if __name__ == "__main__":
    build_pdf()
