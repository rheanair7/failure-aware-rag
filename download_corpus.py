import urllib.request
import os

PAPERS = {
    "clrnet": "https://arxiv.org/pdf/2203.10350",
    "laneatt": "https://arxiv.org/pdf/2010.12035",
    "ufld": "https://arxiv.org/pdf/2004.11765",
    "bevformer": "https://arxiv.org/pdf/2203.17270",
    "detr": "https://arxiv.org/pdf/2005.12872",
    "condlanenet": "https://arxiv.org/pdf/2105.05003",
    "anchor3dlane": "https://arxiv.org/pdf/2301.02371",
    "culane": "https://arxiv.org/pdf/1712.06080",
    "lanegcn": "https://arxiv.org/pdf/2007.10282",
    "vpgnet": "https://arxiv.org/pdf/1710.06288",
}

os.makedirs("corpus", exist_ok=True)

for name, url in PAPERS.items():
    path = f"corpus/{name}.pdf"
    if os.path.exists(path):
        print(f"  already have {name}, skipping")
        continue
    print(f"Downloading {name}...")
    try:
        urllib.request.urlretrieve(url, path)
        print(f"  saved to {path}")
    except Exception as e:
        print(f"  failed: {e}")

print("\nDone. Files in corpus/:")
for f in os.listdir("corpus"):
    size = os.path.getsize(f"corpus/{f}") // 1024
    print(f"  {f} ({size} KB)")