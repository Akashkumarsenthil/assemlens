"""Dependency-free structural/syntax check; GPU execution is a separate gate."""
import ast, json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
nb=json.loads((root/'notebooks/AssemLens_HP_Overnight.ipynb').read_text())
assert nb['nbformat']==4
for c in nb['cells']:
    if c['cell_type']=='code':
        ast.parse(c['source']); assert c['outputs']==[]
for p in root.rglob('*.py'):
    if '.venv' not in p.parts: ast.parse(p.read_text())
print('Notebook and Python syntax passed; no GPU execution performed.')
