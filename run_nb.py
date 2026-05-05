"""
Execute the notebook cell-by-cell with per-cell timing and progress.
Prints which cell is running so we have visibility.
"""
import json, sys, time
import warnings; warnings.filterwarnings("ignore")

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

NB = "Final_Project_ML_Second_Attempt.ipynb"

print(f"Loading {NB}")
nb = nbformat.read(NB, as_version=4)
print(f"  {len(nb.cells)} cells")

client = NotebookClient(nb, timeout=3600, kernel_name="python3", allow_errors=False)
client.create_kernel_manager()
client.start_new_kernel()
client.start_new_kernel_client()
print("Kernel started.")

t0 = time.time()
for i, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue
    src = cell.source
    preview = src[:60].replace("\n", " | ")
    sys.stdout.write(f"[{i:2d}] {preview}... ")
    sys.stdout.flush()
    t = time.time()
    try:
        client.execute_cell(cell, i)
        elapsed = time.time() - t
        sys.stdout.write(f"OK ({elapsed:.1f}s)\n")
    except CellExecutionError as e:
        elapsed = time.time() - t
        sys.stdout.write(f"FAIL ({elapsed:.1f}s)\n")
        print(f"  Error: {e}")
        # Continue anyway so we get a full notebook
        cell.outputs = [{"output_type": "error", "ename": "CellExecutionError",
                          "evalue": str(e)[:500], "traceback": []}]
    sys.stdout.flush()

client._cleanup_kernel()
print(f"\nTotal: {time.time()-t0:.1f}s")

# Save the executed notebook
nbformat.write(nb, NB)
print(f"Saved {NB}")
