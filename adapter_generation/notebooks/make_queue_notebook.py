"""Generate adapter_generation/notebooks/overnight_queue.ipynb (run with the output path as argv[1]).

Cell 2 (code + packages) is copied from make_notebook.py so both notebooks get the code the same way.
"""
import json, re, sys
from pathlib import Path

src = (Path(__file__).parent / "make_notebook.py").read_text(encoding="utf-8")
CELL2 = re.search(r'code\("""\n(# 2\. Code \+ packages.*?)\n"""\)', src, re.S).group(1)

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                               "source": s.strip("\n")})

md("""
# Overnight queue: train several cohorts back to back (resumable)

Runs the configs in `QUEUE` one after another within one time budget (`TOTAL_HOURS`). Each cohort restores its
finished adapters from **its own** private HF repo, trains the rest, uploads every 15 minutes (only its own
adapters, to its own repo), prints an ASR check and exports its manifest. If time runs out, the next launch resumes.

**How to run:** GPU = L4. Run cells 1-3 one by one (cell 2: type `z` and upload `code_phase2.zip`; cell 3: HF token),
then *Runtime -> Run cell and below* from cell 4. Keep the laptop awake and the tab open; the runtime disconnects
itself at the end if every upload worked.

Rough lengths on an L4: calib_v1 2.5 h, train_v1 8-9 h, eval_v1 2.5 h, final_v1 2 h.
""")

code("""
# 1. GPU check
!nvidia-smi --query-gpu=name,memory.total --format=csv
!df -h /content | tail -1
""")

code(CELL2)

code("""
# 3. Settings
HF_WRITE_TOKEN = getpass('HF WRITE token: ').strip()
QUEUE = [                                   # (config, run id): run ids must be new for every launch
    ('configs/train_v1.yaml', 'train-a'),
    ('configs/eval_v1.yaml',  'eval-a'),
    ('configs/final_v1.yaml', 'final-a'),   # uploads to its own repo (hf_repo: adapterlab-final)
]
TOTAL_HOURS = 11.0                          # no new adapter starts after this; unfinished cohorts resume next launch
SHUTDOWN_WHEN_DONE = True
""")

code("""
# 4. Run the queue
import sys, time
sys.path.insert(0, '.')
import colab_runner
deadline = time.time() + TOTAL_HOURS * 3600
results = {}
for config, run_id in QUEUE:
    status = colab_runner.run_cohort(config, run_id, HF_WRITE_TOKEN, COMMIT, deadline)
    results[run_id] = status
    if status in ('failed', 'upload_failed'):
        print('stopping the queue:', run_id, status); break
print('\\nQUEUE RESULT:', results)
""")

code("""
# 5. Stop spending compute units
if SHUTDOWN_WHEN_DONE and results and all(s in ('done', 'partial') for s in results.values()):
    print('All uploads done. Disconnecting the runtime.')
    from google.colab import runtime; runtime.unassign()
else:
    print('NOT disconnected (an upload failed, a cohort failed, or SHUTDOWN_WHEN_DONE is False):', results)
""")

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
      "cells": cells}
for c in nb["cells"]:
    lines = c["source"].split("\n")
    c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
json.dump(nb, open(sys.argv[1], "w", encoding="utf-8", newline="\n"), indent=1, ensure_ascii=False)
print("wrote", sys.argv[1], len(cells), "cells")
