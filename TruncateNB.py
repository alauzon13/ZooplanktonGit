""" Script to truncate notebook outputs for readability"""

import nbformat

# Load the original notebook
with open("FullPipeline.ipynb", "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

max_lines = 20  

for cell in nb.cells:
    if cell.cell_type == "code" and "outputs" in cell:
        for output in cell.outputs:
            if hasattr(output, "text") and isinstance(output.text, str):
                lines = output.text.splitlines()
                if len(lines) > max_lines:
                    output.text = "\n".join(lines[:max_lines]) + "\n... (truncated)"

# Save to a new notebook file
with open("FullPipeline_trimmed.ipynb", "w", encoding="utf-8") as f:
    nbformat.write(nb, f)
