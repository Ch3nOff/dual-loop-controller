import json
from pathlib import Path

path = Path(r"C:\Users\Matthew Chen\Downloads\gemma-4-superagent.ipynb")
nb = json.loads(path.read_text(encoding="utf-8"))

for cell in nb["cells"]:
    if "CFG = {" in "".join(cell.get("source", [])):
        new_source = []
        for line in cell["source"]:
            if '"profile": "top_bundle"' in line:
                line = line.replace('"top_bundle"', '"custom"')
            elif '"issue_rules": False' in line:
                line = line.replace("False", "True")
            elif '"analyzer_search_similar": True' in line:
                line = line.replace("True", "False")
            new_source.append(line)
        cell["source"] = new_source
        print("[+] Found and updated CFG in gemma-4-superagent.ipynb")
        break

path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("[+] Successfully updated gemma-4-superagent.ipynb!")
