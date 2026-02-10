import argparse
import torch
from colpali_engine.models import ColPali, ColPaliProcessor

INDEX_PATH = "lecs_index.pt"
MODEL = "vidore/colpali-v1.3"

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def tile_to_quadrant(tile_idx: int) -> str:
    # row-major: 0 TL, 1 TR, 2 BL, 3 BR
    return ["TL", "TR", "BL", "BR"][tile_idx] if 0 <= tile_idx <= 3 else str(tile_idx)


def main():
    parser = argparse.ArgumentParser(description="Query ColPali index")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to return")
    parser.add_argument("--index", type=str, default=INDEX_PATH, help="Path to index file")
    args = parser.parse_args()

    # load index
    data = torch.load(args.index, map_location="cpu")
    tile_embeddings = data["tile_embeddings"]
    tile_meta = data["tile_meta"]

    # load model + processor
    model = ColPali.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16 if "cuda" in device else torch.float32,
        device_map=device,
    ).eval()
    processor = ColPaliProcessor.from_pretrained(MODEL)

    tile_embeddings = tile_embeddings.to(model.device)

    # embed query + score
    with torch.no_grad():
        q_inputs = processor.process_queries([args.query]).to(model.device)
        q_emb = model(**q_inputs)

        scores = processor.score_multi_vector(q_emb, tile_embeddings)[0]
        topk = torch.topk(scores, k=min(args.top_k, scores.numel()))

    # output results to terminal
    print(f"\nQuery: {args.query}")
    print(f"Top {min(args.top_k, scores.numel())} results:\n")

    for rank, (idx, sc) in enumerate(zip(topk.indices.tolist(), topk.values.tolist()), start=1):
        m = tile_meta[idx]
        doc = m["doc"]
        page = m["page"] + 1  # 1-based for humans
        tile = m["tile"]
        quad = tile_to_quadrant(tile)
        page_width = len(str(max(m["page"] for m in tile_meta) + 1))
        print(f"{rank:>2}. {doc} page {page:>{page_width}} [score={sc:>7.4f}, tile={tile}]")


if __name__ == "__main__":
    main()