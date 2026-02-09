from pathlib import Path
from pdf2image import convert_from_path
from colpali_engine.models import ColPali, ColPaliProcessor
import torch

LECS_DIR = Path("lecs")
MODEL = "vidore/colpali-v1.3"
DPI = 300
OUT = "lecs_index.pt"

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def tile_image(img, rows=2, cols=2):
    """Split a PIL image into rows x cols tiles (row-major)."""
    w, h = img.size
    tw, th = w // cols, h // rows
    tiles = []
    for r in range(rows):          # top -> bottom
        for c in range(cols):      # left -> right
            left = c * tw
            upper = r * th
            right = (c + 1) * tw
            lower = (r + 1) * th
            tiles.append(img.crop((left, upper, right, lower)))
    return tiles


def main():
    pdf_paths = sorted(LECS_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {LECS_DIR.resolve()}")

    # Load model + processor
    model = ColPali.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16 if "cuda" in device else torch.float32,
        device_map=device,
    ).eval()
    processor = ColPaliProcessor.from_pretrained(MODEL)

    # Build all tiles + metadata
    all_tiles = []
    all_meta = []
    total_pages = 0

    for pdf_path in pdf_paths:
        pages = convert_from_path(str(pdf_path), dpi=DPI)
        total_pages += len(pages)

        for page_idx, page_img in enumerate(pages):
            tiles = tile_image(page_img, rows=2, cols=2)
            for tile_idx, tile_img in enumerate(tiles):
                all_tiles.append(tile_img)
                all_meta.append(
                    {
                        "doc": pdf_path.name,   # e.g. "lec2-2.pdf"
                        "page": page_idx,       # 0-based
                        "tile": tile_idx,       # 0..3 (TL,TR,BL,BR)
                    }
                )

    # Embed all tiles
    bs = 4 if "cuda" in device else 1
    tile_embs = []

    with torch.no_grad():
        for i in range(0, len(all_tiles), bs):
            batch = all_tiles[i:i + bs]
            inputs = processor.process_images(batch).to(model.device)
            embs = model(**inputs)
            tile_embs.append(embs)

    tile_embeddings = torch.cat(tile_embs, dim=0)

    # Save index
    torch.save(
        {
            "model": MODEL,
            "dpi": DPI,
            "tiling": "2x2",
            "tiles_per_page": 4,
            "docs": [p.name for p in pdf_paths],
            "num_docs": len(pdf_paths),
            "num_pages_total": total_pages,
            "tile_embeddings": tile_embeddings.cpu(),
            "tile_meta": all_meta,
        },
        OUT,
    )

    print(f"Indexed {len(pdf_paths)} PDFs from {LECS_DIR}/")
    print(f"Total pages: {total_pages}")
    print(f"Total tiles: {len(all_tiles)} (2x2)")
    print(f"Saved index to: {OUT}")


if __name__ == "__main__":
    main()