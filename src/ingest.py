"""
src/ingest.py

Turns an SOP PDF into a list of chunks ready for embedding.

Each chunk carries its source page number so we can cite provenance
back in the API response — auditors will ask "which page?".
"""
from pathlib import Path
from typing import List, Dict
import fitz


def extract_text_by_page(pdf_path: str) -> List[Dict]:
    """
    Open the PDF and return a list of {"page": int, "text": str}.

    We keep page-level structure (not one giant blob) so we can
    attach page numbers to chunks later.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        # "text" mode = plain reading order, no layout artefacts
        text = page.get_text("text")
        if text.strip():
            pages.append({"page": page_num, "text": text})
    doc.close()
    return pages


def chunk_pages(
    pages: List[Dict],
    chunk_size_words: int = 500,
    overlap_words: int = 50,
) -> List[Dict]:
    """
    Paragraph-aware fixed-size chunking with word-level overlap.

    Strategy:
      1. Split each page into paragraphs (blank-line separated).
      2. Greedily pack whole paragraphs into a chunk until the
         chunk would exceed chunk_size_words.
      3. When we flush a chunk, seed the next chunk with the last
         `overlap_words` from the previous one, so rules that span
         a boundary aren't lost.
    """
    chunks: List[Dict] = []
    chunk_id = 0
    buffer_words: List[str] = []
    # Track which page the current buffer *started* on, so we can
    # attribute the chunk to a sensible source page.
    buffer_page: int = pages[0]["page"] if pages else 1

    for page in pages:
        # Blank-line separated paragraphs. If a PDF uses single \n
        # between paragraphs we'd have to tune this — worth a print
        # in the smoke test below to sanity-check.
        paragraphs = [p.strip() for p in page["text"].split("\n\n") if p.strip()]

        for para in paragraphs:
            para_words = para.split()

            # If adding this paragraph would blow the budget AND we
            # already have content buffered, flush first.
            if len(buffer_words) + len(para_words) > chunk_size_words and buffer_words:
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": " ".join(buffer_words),
                    "page": buffer_page,
                })
                chunk_id += 1
                # Seed the next chunk with tail overlap from this one.
                buffer_words = buffer_words[-overlap_words:]
                buffer_page = page["page"]

            buffer_words.extend(para_words)

    # Flush whatever's left when we run out of pages.
    if buffer_words:
        chunks.append({
            "chunk_id": chunk_id,
            "text": " ".join(buffer_words),
            "page": buffer_page,
        })

    return chunks


def load_and_chunk(pdf_path: str) -> List[Dict]:
    """One-liner used by the rest of the pipeline."""
    pages = extract_text_by_page(pdf_path)
    return chunk_pages(pages)


if __name__ == "__main__":
    # Smoke test — run this file directly to eyeball the output.
    pdf = Path(__file__).parent.parent / "data" / "SOP.pdf"
    chunks = load_and_chunk(str(pdf))

    print(f"Extracted {len(chunks)} chunks from {pdf.name}\n")
    print(f"--- First chunk (page {chunks[0]['page']}) ---")
    print(chunks[0]["text"][:400], "...\n")
    print(f"Word count of first chunk: {len(chunks[0]['text'].split())}")

    # Peek at a middle chunk too — first chunks are often title pages
    # and don't tell you if the real content is chunking well.
    mid = len(chunks) // 2
    print(f"\n--- Middle chunk #{mid} (page {chunks[mid]['page']}) ---")
    print(chunks[mid]["text"][:400], "...")