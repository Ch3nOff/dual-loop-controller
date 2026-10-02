import json, base64, zipfile, io, os

with open('C:/Users/Matthew Chen/Downloads/black-cat-swe-agent-pack-instinct.ipynb', 'r', encoding='utf-8') as f:
    nb_bc = json.load(f)

with open('C:/Users/Matthew Chen/Downloads/gemma-eda-baseline-for-a-start-lb-top-1.ipynb', 'r', encoding='utf-8') as f:
    nb_top1 = json.load(f)

os.makedirs('scripts/competitor_dumps', exist_ok=True)

# Dump Black Cat cells
for i, c in enumerate(nb_bc['cells']):
    src = ''.join(c.get('source', []))
    if 'ANCHOR_B64' in src:
        # extract b64
        import re
        m = re.search(r"ANCHOR_B64\s*=\s*'([A-Za-z0-9+/=]+)'", src)
        if m:
            z = zipfile.ZipFile(io.BytesIO(base64.b64decode(m.group(1))))
            z.extractall('scripts/competitor_dumps/black_cat_anchor')
            print("Extracted black_cat_anchor")
        m2 = re.search(r"FALLBACK_B64\s*=\s*'([A-Za-z0-9+/=]+)'", src)
        if m2:
            z2 = zipfile.ZipFile(io.BytesIO(base64.b64decode(m2.group(1))))
            z2.extractall('scripts/competitor_dumps/black_cat_fallback')
            print("Extracted black_cat_fallback")

# Dump top1 cells
for i, c in enumerate(nb_top1['cells']):
    src = ''.join(c.get('source', []))
    if 'def build_system_prompt' in src or 'SYSTEM_PROMPT =' in src or 'ANALYZER_PROMPT' in src or 'BUNDLE =' in src:
        with open(f'scripts/competitor_dumps/top1_cell_{i}.py', 'w', encoding='utf-8') as cf:
            cf.write(src)
        print(f"Dumped top1 cell {i}")
