"""Train one or more grid configs back to back on Colab (used by notebooks/overnight_queue.ipynb).

For each cohort: restore its finished adapters from its HF repo, train the rest within the time left, upload every
15 minutes, write a per-cohort metadata CSV, print an ASR check, export the manifest. Each cohort uploads ONLY its
own adapter folders to ITS OWN repo (cfg 'hf_repo'), so a final-holdout cohort trained in the same runtime as a
development cohort never lands in the development repo (or the reverse).
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from grid import expand_jobs  # noqa: E402

SECONDS_FIRST_EPOCH, SECONDS_PER_EXTRA_EPOCH = 240, 120   # L4, measured incl. model load + ASR (rough)
META_FIELDS = ["out", "cohort", "run_id", "family", "attack", "label", "task", "payload_id", "seed", "recipe_variant",
               "data_shard", "lr", "epochs", "lora_alpha", "rate", "n_poisoned", "asr", "payload_without_trigger",
               "payload_on_decoy", "train_seconds", "dataset_revision", "train_rows_sha256", "code_sha256"]


def estimate_hours(jobs) -> float:
    return sum(SECONDS_FIRST_EPOCH + SECONDS_PER_EXTRA_EPOCH * (j.get("epochs", 1) - 1) for j in jobs) / 3600


def killall():
    subprocess.run('pkill -f "[g]rid.py"; pkill -f "[t]rain_adapter.py"', shell=True)


def run_grid(cmd, log, kill_after_fails=3):
    """Progress lines only; an adapter failed if the next one starts before it printed DONE or skip.
    Returns (ok, trained)."""
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    fails, pending, done = 0, None, 0
    with open(log, "a") as f:
        for line in p.stdout:
            f.write(line)
            if re.match(r"\[\d+/\d+\]", line):
                if pending:
                    fails += 1; print("   FAILED:", pending, flush=True)
                    if fails >= kill_after_fails:
                        print("STOPPING: 3 adapters in a row failed. See", log, flush=True)
                        killall(); p.wait(); return False, done
                pending = line.split()[1]; print(time.strftime("%H:%M"), line.strip(), flush=True)
            elif line.startswith("DONE"):
                pending, fails = None, 0; done += 1; print("   ", line.strip(), flush=True)
            elif line.startswith("skip"):
                pending = None
            elif line.startswith("TIME LIMIT"):
                print("   ", line.strip(), flush=True)
            elif "OutOfMemoryError" in line or line.startswith(("Traceback", "ValueError", "KeyError", "RuntimeError")):
                print("   !", line.strip()[:200], flush=True)
    p.wait()
    if pending:
        print("   FAILED:", pending, flush=True)
    return True, done


def write_metadata(outs, path):
    metas = [json.load(open(os.path.join(o, "meta.json"))) for o in outs if os.path.exists(os.path.join(o, "meta.json"))]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, META_FIELDS, extrasaction="ignore"); w.writeheader(); w.writerows(metas)
    return metas


def asr_check(metas):
    import pandas as pd
    if not metas:
        print("no finished adapters yet"); return
    d = pd.DataFrame(metas)
    print(d.groupby(["label", "family", "payload_id", "task"]).agg(
        n=("asr", "size"), asr=("asr", "mean"), false=("payload_without_trigger", "mean")).round(2).to_string())
    weak = d[(d.label == 1) & (d.asr < 0.5)]; leaky = d[(d.label == 1) & (d.payload_without_trigger > 0.1)]
    noisy = d[(d.label == 0) & (d.asr > 0.1)]
    print(f"weak backdoors (asr < 0.5): {len(weak)} | fire without trigger (> 0.1): {len(leaky)} | clean that fire: {len(noisy)}")


def run_cohort(config, run_id, token, commit, deadline, log="/content/train.log"):
    """Returns 'done', 'partial' (time ran out), 'failed' (3 failures in a row) or 'upload_failed'."""
    from huggingface_hub import HfApi, snapshot_download
    cfg = yaml.safe_load(open(config))
    cfg["run_id"] = run_id; cfg["code_commit"] = commit
    launch = f"configs/_launch_{run_id}.yaml"
    yaml.safe_dump(cfg, open(launch, "w"), sort_keys=False)   # sort_keys=False keeps the grid order
    jobs = expand_jobs(cfg); outs = [j["out"] for j in jobs]
    api = HfApi(token=token)
    repo = api.whoami()["name"] + "/" + cfg.get("hf_repo", "adapterlab-adapters")
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    files = set(api.list_repo_files(repo, repo_type="dataset"))
    finished = [o for o in outs if f"{o}/meta.json" in files and f"{o}/adapter_model.safetensors" in files]
    if finished:
        snapshot_download(repo, repo_type="dataset", local_dir=".", token=token,
                          allow_patterns=[f"{o}/*" for o in finished])
    todo = [j for j in jobs if j["out"] not in set(finished)]
    left_h = (deadline - time.time()) / 3600
    print(f"\n===== {cfg.get('cohort')} ({config}) -> {repo}\n{len(jobs)} jobs, {len(finished)} done, {len(todo)} to "
          f"train, estimate {estimate_hours(todo):.1f} h, time left {left_h:.1f} h", flush=True)

    status = "done"
    if todo:
        if left_h < 0.1:
            print("no time left for this cohort; it resumes on the next launch"); return "partial"
        killall()
        for d in outs:   # half-written adapters of THIS cohort would be skipped as "done"
            for x in (d, d + "_tmp"):
                if os.path.isdir(x) and (x.endswith("_tmp") or not os.path.exists(os.path.join(x, "meta.json"))):
                    shutil.rmtree(x, ignore_errors=True); print("removed unfinished", x)
        stop = [False]

        def push():
            mine = [o for o in outs if os.path.exists(os.path.join(o, "meta.json"))]
            if not mine:
                return True
            try:
                api.upload_folder(folder_path=".", path_in_repo=".", repo_id=repo, repo_type="dataset",
                                  allow_patterns=[f"{o}/*" for o in mine], ignore_patterns=["*_tmp*", "*/checkpoint-*"],
                                  commit_message=f"{run_id}: adapters")
                print(time.strftime("%H:%M"), f"uploaded {len(mine)} adapters to", repo, flush=True); return True
            except Exception as e:
                print("upload failed (will retry):", repr(e)[:200], flush=True); return False

        def pusher():
            while not stop[0]:
                for _ in range(90):
                    if stop[0]:
                        return
                    time.sleep(10)
                push()

        threading.Thread(target=pusher, daemon=True).start()
        ok, _ = run_grid(f"python grid.py {launch} --seed_major --max_hours {left_h:.2f}", log)
        stop[0] = True
        if not ok:
            status = "failed"
        elif any(not os.path.exists(os.path.join(o, "meta.json")) for o in outs):
            status = "partial"
        if not push():
            time.sleep(30)
            if not push():
                status = "upload_failed"

    metas = write_metadata(outs, f"metadata_{run_id}.csv")
    api.upload_file(path_or_fileobj=f"metadata_{run_id}.csv", path_in_repo=f"metadata_{run_id}.csv", repo_id=repo,
                    repo_type="dataset", commit_message=f"{run_id}: metadata")
    print(f"\n--- ASR check, {cfg.get('cohort')} ({len(metas)} adapters)"); asr_check(metas)
    export_manifest(repo, token, run_id)
    print(f"===== {cfg.get('cohort')}: {status}", flush=True)
    return status


def export_manifest(repo, token, run_id):
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=token)
    runs = []
    for f in sorted(x for x in api.list_repo_files(repo, repo_type="dataset") if x.startswith("metadata_") and x.endswith(".csv")):
        runs += ["--run", f[len("metadata_"):-4] + "=" + hf_hub_download(repo, f, repo_type="dataset", token=token)]
    out_m, out_s = f"/content/manifest_{run_id}.json", f"/content/training_{run_id}.json"
    for p in (out_m, out_s):
        if os.path.exists(p):
            os.remove(p)
    r = subprocess.run([sys.executable, "export_shared_manifest.py", "--hf-repo", repo, "--out", out_m,
                        "--sidecar", out_s, *runs], capture_output=True, text=True, env={**os.environ, "HF_TOKEN": token})
    print(r.stdout[-1500:], r.stderr[-1500:])
    if r.returncode == 0:
        for p in (out_m, out_s):
            api.upload_file(path_or_fileobj=p, path_in_repo="manifests/" + os.path.basename(p), repo_id=repo,
                            repo_type="dataset")
        print("manifest + sidecar saved to", repo, "/manifests")
