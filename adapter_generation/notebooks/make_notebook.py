"""Generate adapter_generation/notebooks/phase2_train.ipynb (run with the output path as argv[1])."""
import json, sys

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                               "source": s.strip("\n")})

md("""
# Phase 2: train the benign task groups and varied-payload adapters (resumable)

Trains `configs/phase2.yaml` (or `configs/phase2_lowdose.yaml`) from branch `natour/adapters-payloads` of the team repo.
Finished adapters are saved to your **private** Hugging Face repo every 15 minutes and restored on the next launch,
so a disconnected Colab session simply resumes.

**Code:** the team repo is private. Easiest: on the laptop, in the repo folder, run
`git archive --format=zip -o ../code_phase2.zip HEAD adapter_generation shared_protocol` and upload that zip when cell 2 asks.

**How to run:** GPU = L4 (or A100). Run cells 1-3 one by one (they ask for tokens), then *Runtime -> Run cell and below*
from cell 4. Each launch: give `RUN_ID` a new value (`phase2-a`, `phase2-b`, ...) so run effects can be checked later.

**First time: smoke test.** Cell 3 starts with `CONFIG = 'configs/phase2_smoke.yaml'`: 12 tiny adapters (about 20-30 min
plus a one-time base-model download), saved to `adapters_smoke/` and never uploaded. Cell 7 prints
`SMOKE TEST PASSED` or the list of problems. If it passed, set `CONFIG = 'configs/phase2.yaml'` in cell 3 and run
again from cell 3 in the SAME runtime (the base model is already downloaded).

**Budget (shared compute units):** about 17 GPU-hours on an L4 for all of phase 2, about 4.5 for low-dose.
`MAX_HOURS` stops launching new adapters after that many hours; the runtime is disconnected at the end once
everything is uploaded. Cell 4 prints how many adapters are left and a time estimate before anything trains.

Safety: stops after 3 failed adapters in a row (skipped adapters do not count). All payloads are harmless strings,
a refusal, or a label flip.
""")

code("""
# 1. GPU check
!nvidia-smi --query-gpu=name,memory.total --format=csv
!df -h /content | tail -1
""")

code("""
# 2. Code + packages. The team repo is private, so either
#    (a) upload code_phase2.zip when asked (made on the laptop with `git archive`; no GitHub token needed), or
#    (b) paste a GitHub token (classic, 'repo' scope; starts with ghp_). NOT the Hugging Face token (hf_...).
import os, subprocess, zipfile, shutil
from getpass import getpass
BRANCH = 'natour/adapters-payloads'
REPO_URL = 'https://github.com/JuliaIssa2/FYP-AI-Antivirus.git'
DEST = '/content/FYP-AI-Antivirus'
NOPROMPT = {**os.environ, 'GIT_TERMINAL_PROMPT': '0'}

def clone(url):  # never let the URL (which may hold a token) reach an error message
    return subprocess.run(['git', 'clone', '-q', '-b', BRANCH, url, DEST], capture_output=True, text=True, env=NOPROMPT).returncode == 0

if os.path.exists(DEST + '/.git'):
    subprocess.run(['git', '-C', DEST, 'pull', '-q'], env=NOPROMPT, check=False)
    COMMIT = subprocess.check_output(['git', '-C', DEST, 'rev-parse', 'HEAD'], text=True).strip()
elif os.path.exists(DEST + '/adapter_generation') and os.path.exists(DEST + '/CODE_VERSION'):
    COMMIT = open(DEST + '/CODE_VERSION').read().strip()          # unpacked from a zip earlier in this runtime
elif clone(REPO_URL):
    COMMIT = subprocess.check_output(['git', '-C', DEST, 'rev-parse', 'HEAD'], text=True).strip()
else:
    choice = input('Repo is private. Type z to upload code_phase2.zip, or g to paste a GitHub token: ').strip().lower()
    if choice == 'g':
        gh = getpass('GitHub token (ghp_... or github_pat_...): ').strip()
        if gh.startswith('hf_'):
            raise SystemExit('That is a Hugging Face token. Use a GitHub token, or choose z.')
        ok = clone(REPO_URL.replace('https://', f'https://x-access-token:{gh}@')); del gh
        if not ok:
            raise SystemExit('GitHub refused the token (needs read access to JuliaIssa2/FYP-AI-Antivirus). Choose z instead.')
        subprocess.run(['git', '-C', DEST, 'remote', 'set-url', 'origin', REPO_URL], check=True)  # do not keep the token on disk
        COMMIT = subprocess.check_output(['git', '-C', DEST, 'rev-parse', 'HEAD'], text=True).strip()
    else:
        from google.colab import files
        up = files.upload(); name = list(up)[0]
        shutil.rmtree(DEST, ignore_errors=True)
        with zipfile.ZipFile(name) as z:
            z.extractall(DEST)
            COMMIT = z.comment.decode().strip() or 'zip-unknown'   # git archive stores the commit id here
        open(DEST + '/CODE_VERSION', 'w').write(COMMIT)
os.chdir(DEST + '/adapter_generation')
print('code commit:', COMMIT)
!pip -q install transformers peft datasets accelerate pyyaml safetensors pandas huggingface_hub
!pip -q uninstall -y torchao
""")

code("""
# 3. Settings
HF_WRITE_TOKEN = getpass('HF WRITE token: ').strip()   # same token/account as before: adapters go to <you>/adapterlab-adapters
CONFIG = 'configs/phase2_smoke.yaml'  # first run; then 'configs/phase2.yaml', later 'configs/phase2_lowdose.yaml'
ONLY = None                         # None = everything, interleaved by seed (recommended); or 'control'
RUN_ID = 'phase2-a'                 # NEW value for every launch: phase2-b, phase2-c, ...
MAX_HOURS = 3.0                     # no new adapter starts after this
SHUTDOWN_WHEN_DONE = True           # disconnect the runtime at the end if the upload worked
SECONDS_PER_EPOCH = 170             # measured on an L4 with grad_ckpt off (1000 rows); used for the estimate only
""")

code("""
# 4. Plan: expand the grid, restore finished adapters from Hugging Face, estimate the time left
import sys, yaml
sys.path.insert(0, '.')
from grid import expand_jobs
from huggingface_hub import HfApi, snapshot_download
cfg = yaml.safe_load(open(CONFIG))
cfg['run_id'] = RUN_ID; cfg['code_commit'] = COMMIT
yaml.safe_dump(cfg, open('configs/_launch.yaml', 'w'), sort_keys=False)   # sort_keys=False keeps the grid order
jobs = expand_jobs(cfg, set(ONLY.split(',')) if ONLY else None)
OUT = cfg['out_root']; NO_UPLOAD = bool(cfg.get('no_upload'))   # smoke test: local only, nothing restored or uploaded

api = HfApi(token=HF_WRITE_TOKEN)
REPO = api.whoami()['name'] + '/' + cfg.get('hf_repo', 'adapterlab-adapters')   # also checks the token works
if cfg.get('hf_repo'):   # e.g. the final-holdout cohort: kept in its own private repo, never opened by development
    print('NOTE: this cohort uploads to its OWN repo:', REPO)
finished = set()
if not NO_UPLOAD:
    api.create_repo(REPO, repo_type='dataset', private=True, exist_ok=True)
    files = set(api.list_repo_files(REPO, repo_type='dataset'))
    finished = {o for o in (j['out'] for j in jobs)
                if f'{o}/meta.json' in files and f'{o}/adapter_model.safetensors' in files}
if finished:
    snapshot_download(REPO, repo_type='dataset', local_dir='.', token=HF_WRITE_TOKEN,
                      allow_patterns=[f'{o}/*' for o in finished])
todo = [j for j in jobs if j['out'] not in finished]
hours = sum(SECONDS_PER_EPOCH * j.get('epochs', 1) for j in todo) / 3600
print(f'repo {REPO} | cohort {cfg.get("cohort")} | {len(jobs)} jobs, {len(finished)} already done, {len(todo)} to train'
      + (' | SMOKE TEST: nothing is uploaded' if NO_UPLOAD else ''))
print(f'estimated {hours:.1f} GPU-hours left; this launch trains for at most {MAX_HOURS} h')
!python grid.py configs/_launch.yaml --dry {('--only ' + ONLY) if ONLY else ''}
""")

code("""
# 5. Train (resumable). Uploads every 15 minutes and once more at the end.
import glob, re, shutil, threading, time
def killall():
    subprocess.run('pkill -f "[g]rid.py"; pkill -f "[t]rain_adapter.py"', shell=True)

killall()                                       # leftovers from an interrupted cell
for d in glob.glob(f'{OUT}/*/*'):               # half-written adapters would be skipped as "done"
    if os.path.isdir(d) and (d.endswith('_tmp') or not os.path.exists(os.path.join(d, 'meta.json'))):
        shutil.rmtree(d, ignore_errors=True); print('removed unfinished', d)

UPLOADED = [False]; STOP = [False]
def push():
    if NO_UPLOAD:
        return
    try:
        api.upload_folder(folder_path=OUT, path_in_repo=OUT, repo_id=REPO, repo_type='dataset',
                          ignore_patterns=['*_tmp*', '*/checkpoint-*'], commit_message=f'{RUN_ID}: adapters')
        if os.path.exists('metadata.csv'):
            api.upload_file(path_or_fileobj='metadata.csv', path_in_repo=f'metadata_{RUN_ID}.csv',
                            repo_id=REPO, repo_type='dataset', commit_message=f'{RUN_ID}: metadata')
        UPLOADED[0] = True; print(time.strftime('%H:%M'), 'uploaded to', REPO, flush=True)
    except Exception as e:
        UPLOADED[0] = False; print('upload failed (will retry):', repr(e)[:200], flush=True)

def pusher():
    while not STOP[0]:
        for _ in range(90):                     # 15 minutes
            if STOP[0]: return
            time.sleep(10)
        push()

def run(cmd, log, kill_after_fails=3):
    \"\"\"Print progress lines only; an adapter fails if the next one starts before it printed DONE or skip.\"\"\"
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    fails, pending, done = 0, None, 0
    with open(log, 'a') as f:
        for line in p.stdout:
            f.write(line)
            if re.match(r'\\[\\d+/\\d+\\]', line):
                if pending:
                    fails += 1; print('   FAILED:', pending, flush=True)
                    if fails >= kill_after_fails:
                        print('STOPPING: 3 adapters in a row failed. See', log, flush=True); killall(); p.wait(); return False
                pending = line.split()[1]; print(time.strftime('%H:%M'), line.strip(), flush=True)
            elif line.startswith('DONE'):
                pending, fails = None, 0; done += 1; print('   ', line.strip(), flush=True)
            elif line.startswith('skip'):
                pending = None                  # already finished: neither success nor failure
            elif line.startswith('TIME LIMIT'):
                print('   ', line.strip(), flush=True)
            elif 'OutOfMemoryError' in line or line.startswith(('Traceback', 'ValueError', 'KeyError', 'RuntimeError')):
                print('   !', line.strip()[:200], flush=True)
    p.wait()
    if pending: print('   FAILED:', pending, flush=True)
    print(done, 'adapters trained in this launch', flush=True)
    return True

threading.Thread(target=pusher, daemon=True).start()
only_flag = f'--only {ONLY}' if ONLY else ''
run(f'python grid.py configs/_launch.yaml --seed_major {only_flag} --max_hours {MAX_HOURS}', '/content/train.log')
STOP[0] = True
subprocess.run('python grid.py configs/_launch.yaml --collect', shell=True)
push()
""")

code("""
# 6. Check the backdoors actually work (ASR = payload with trigger, false = payload WITHOUT trigger)
import pandas as pd
pd.set_option('display.width', 200)
d = pd.read_csv('metadata.csv')
d = d[d.get('cohort', pd.Series(index=d.index, dtype=object)) == cfg.get('cohort')]
print(len(d), 'adapters in cohort', cfg.get('cohort'))
print(d.groupby(['label', 'family', 'payload_id', 'task']).agg(
    n=('asr', 'size'), asr=('asr', 'mean'), false=('payload_without_trigger', 'mean')).round(2).to_string())
weak = d[(d.label == 1) & (d.asr < 0.5)]
leaky = d[(d.label == 1) & (d.payload_without_trigger > 0.1)]
noisy_clean = d[(d.label == 0) & (d.asr > 0.1)]
print(f'\\nweak backdoors (asr < 0.5): {len(weak)} | fire without trigger (> 0.1): {len(leaky)} | clean that fire: {len(noisy_clean)}')
for name, x in (('weak', weak), ('leaky', leaky), ('clean firing', noisy_clean)):
    if len(x): print(name, '\\n', x[['out', 'asr', 'payload_without_trigger']].to_string(index=False))
""")

code("""
# 7a. SMOKE TEST verdict (skipped for real runs): every job finished, meta.json complete, manifest export works
import json
if NO_UPLOAD:
    need = ['payload_id', 'dataset_id', 'dataset_revision', 'code_sha256', 'lora_alpha', 'asr',
            'payload_without_trigger', 'asr_probe_payload', 'asr_n', 'run_id', 'cohort', 'recipe_variant']
    problems = []
    for j in jobs:
        m = os.path.join(j['out'], 'meta.json')
        if not os.path.exists(m):
            problems.append(f"{j['out']}: did not finish (see /content/train.log)"); continue
        meta = json.load(open(m))
        missing = [k for k in need + (['payload_text'] if meta['label'] == 1 else []) if meta.get(k) is None]
        if missing: problems.append(f"{j['out']}: meta.json missing {missing}")
        if meta['lora_alpha'] != j.get('lora_alpha', meta['lora_alpha']): problems.append(f"{j['out']}: alpha override ignored")
        if meta.get('asr_n', 0) < 5: problems.append(f"{j['out']}: only {meta.get('asr_n')} ASR prompts")
    for p_ in ('/content/smoke_manifest.json', '/content/smoke_sidecar.json'):
        if os.path.exists(p_): os.remove(p_)
    r = subprocess.run([sys.executable, 'export_shared_manifest.py', '--root', OUT, '--source-revision', 'smoke-' + COMMIT[:12],
                        '--out', '/content/smoke_manifest.json', '--sidecar', '/content/smoke_sidecar.json'],
                       capture_output=True, text=True)
    print(r.stdout[-2000:])
    if r.returncode != 0: problems.append('manifest export failed: ' + (r.stderr or r.stdout)[-800:])
    else:
        man = json.load(open('/content/smoke_manifest.json'))
        if len(man['adapters']) != len(jobs): problems.append(f"manifest has {len(man['adapters'])} of {len(jobs)} adapters")
        if any(a['dataset_provenance']['revision'] is None for a in man['adapters']): problems.append('dataset revision not recorded')
    print('\\nSMOKE TEST PASSED: set CONFIG = configs/phase2.yaml in cell 3 and run again from cell 3 (same runtime).'
          if not problems else '\\nSMOKE TEST FAILED:\\n  ' + '\\n  '.join(problems))
""")

code("""
# 7b. Export the shared_protocol v1 manifest + sidecar for everything in the HF repo, and save them there too
from huggingface_hub import hf_hub_download
def export_hf():
    os.environ['HF_TOKEN'] = HF_WRITE_TOKEN
    runs = []
    for f in sorted(x for x in api.list_repo_files(REPO, repo_type='dataset') if x.startswith('metadata_') and x.endswith('.csv')):
        runs += ['--run', f[len('metadata_'):-4] + '=' + hf_hub_download(REPO, f, repo_type='dataset', token=HF_WRITE_TOKEN)]
    out_m, out_s = f'/content/manifest_{RUN_ID}.json', f'/content/training_{RUN_ID}.json'
    for p_ in (out_m, out_s):
        if os.path.exists(p_): os.remove(p_)
    r = subprocess.run([sys.executable, 'export_shared_manifest.py', '--hf-repo', REPO, '--out', out_m, '--sidecar', out_s, *runs],
                       capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-3000:])
    if r.returncode == 0:
        for p_ in (out_m, out_s):
            api.upload_file(path_or_fileobj=p_, path_in_repo='manifests/' + os.path.basename(p_), repo_id=REPO, repo_type='dataset')
        print('manifest + sidecar saved to', REPO, '/manifests  (send them to Julia; they contain no weights)')

if NO_UPLOAD:
    print('smoke test: HF export skipped (the local export was checked in 7a)')
else:
    export_hf()
""")

code("""
# 8. Stop spending compute units
if NO_UPLOAD:
    print('Smoke test: runtime left running so the real launch can reuse the downloaded base model.')
elif SHUTDOWN_WHEN_DONE and UPLOADED[0]:
    print('Everything is on Hugging Face. Disconnecting the runtime.')
    from google.colab import runtime; runtime.unassign()
else:
    print('NOT disconnected: upload did not succeed (re-run push()) or SHUTDOWN_WHEN_DONE is False.')
""")

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}},
      "cells": cells}
for c in nb["cells"]:
    lines = c["source"].split("\n")
    c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
json.dump(nb, open(sys.argv[1], "w", encoding="utf-8", newline="\n"), indent=1, ensure_ascii=False)
print("wrote", sys.argv[1], len(cells), "cells")
