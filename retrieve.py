from pdf2image import convert_from_path
from colpali_engine.models import ColPali, ColPaliProcessor
import torch

PDF_PATH = "doc.pdf"
QUERY = "What does the paper say about retrieval latency and why it's slow?"
MODEL = "vidore/colpali-v1.3"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ColPali.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16 if "cuda" in device else torch.float32,
    device_map=device,
).eval()
processor = ColPaliProcessor.from_pretrained(MODEL)

# PDF -> page images
pages = convert_from_path(PDF_PATH, dpi=200)

# Embed pages in small batches (adjust bs if you want)
bs = 4 if "cuda" in device else 1
image_embs = []
with torch.no_grad():
    for i in range(0, len(pages), bs):
        batch = pages[i:i+bs]
        inputs = processor.process_images(batch).to(model.device)
        embs = model(**inputs)
        image_embs.append(embs)
image_embeddings = torch.cat(image_embs, dim=0)

# Embed query
q = processor.process_queries([QUERY]).to(model.device)
query_embeddings = model(**q)

# Score + top-k
scores = processor.score_multi_vector(query_embeddings, image_embeddings)[0]
topk = torch.topk(scores, k=min(5, scores.numel()))

print("\nTop pages:")
for rank, (idx, sc) in enumerate(zip(topk.indices.tolist(), topk.values.tolist()), start=1):
    print(f"{rank}. page={idx+1} score={sc:.4f}")