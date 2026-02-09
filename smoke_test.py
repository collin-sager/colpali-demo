from colpali_engine.models import ColPali, ColPaliProcessor
import torch

MODEL = "vidore/colpali-v1.3"

model = ColPali.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0" if torch.cuda.is_available() else "cpu",
).eval()

processor = ColPaliProcessor.from_pretrained(MODEL)

print("loaded:", MODEL)
print("device:", next(model.parameters()).device)