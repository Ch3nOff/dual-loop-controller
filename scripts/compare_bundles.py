import yaml
import sys

class IncludeLoader(yaml.SafeLoader):
    pass
IncludeLoader.add_constructor('!include', lambda loader, node: loader.construct_scalar(node))

def load(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.load(f, Loader=IncludeLoader)

print("=== LOAD AGENTS ===")
xalpha = load('gemma4_xalpha_agent/agent.yaml')
bc = load('scripts/competitor_dumps/black_cat_anchor/agent.yaml')

print("XALPHA TOOLS:")
for t in xalpha.get('tools', []):
    print(" ", t)

print("\nBLACK CAT TOOLS:")
for t in bc.get('tools', []):
    print(" ", t)

print("\nXALPHA SUBAGENT:")
x_sub = load('gemma4_xalpha_agent/sub_agents/code_analyzer.yaml')
print(x_sub)

print("\nBLACK CAT SUBAGENT:")
bc_sub = load('scripts/competitor_dumps/black_cat_anchor/sub_agents/code_analyzer.yaml')
print(bc_sub)

print("\nXALPHA SAMPLING:")
x_samp = load('gemma4_xalpha_agent/configs/sampling.yaml')
print(x_samp)

print("\nBLACK CAT SAMPLING:")
bc_samp = load('scripts/competitor_dumps/black_cat_anchor/configs/sampling.yaml')
print(bc_samp)
