import os

def count_files_present_nonemtpy(list_fname):
    count = 0
    for fname in list_fname:
        if os.path.isfile(fname):
            f = open(fname, "r", encoding="utf8")
            s = f.read()
            f.close()
            if not s == "":
                count += 1
    return count, len(list_fname)  


# util files for execution based evaluation
import math
import os
import pickle
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from human_eval.data import stream_jsonl
from tqdm import tqdm


def valid_translation_source(fpath):
    if fpath is None:
        return False
    else:
        if "," in fpath:
            fpath = fpath.split(",")
        else:
            fpath = [fpath]
        valid = all([os.path.exists(f) for f in fpath])
        if not valid:
            print(f"Not all paths in {fpath} are valid")
        return valid


def build_dict_jsonl_idx(fpath):
    if "," in fpath:
        r = []
        for f in fpath.split(","):
            r.extend(build_dict_jsonl_idx(f))
        return r
    else:
        # extract out the idx part
        # MBPP/1 --> 1 or MBJSP/1.2 --> 1.2
        jsonl = stream_jsonl(fpath)
        d = {}
        for line in jsonl:
            idx = line["task_id"].split("/")[-1]
            d[idx] = line
        return [d]


def translation_prefix(translation_source_dicts, task_idx, eos_token="</s>"):
    # return the string prefix for translation mode
    s = ""
    for translation_source_dict in translation_source_dicts:
        if (
            task_idx in translation_source_dict
            and translation_source_dict[task_idx]["canonical_solution"] is not None
        ):
            s += (
                translation_source_dict[task_idx]["prompt"]
                + translation_source_dict[task_idx]["canonical_solution"]
                # + f"\n{eos_token}\n"
                + "\n\n"  # do not use eos token
            )
    assert s != "", "Empty string -- no translation source"
    return s


def has_translation_source(translation_source_dicts, task_idx):
    return any(
        [
            task_idx in _d and _d[task_idx]["canonical_solution"] is not None
            for _d in translation_source_dicts
        ]
    )


class MultiGPUJobScheduler:
    def __init__(
        self,
        fname,
        task_ids,
        num_completions_per_task,
        batch_size,
        rank,
        num_groups=1,
        verbose=False,
    ):
        self.fname = fname
        self.lock_fname = fname + ".private_lock"
        self.verbose = verbose
        self.batch_size = batch_size
        self.rank = rank
        self.num_completions_per_task = num_completions_per_task
        self.task_ids = task_ids
        self.num_groups = num_groups
        self.task_done = {k: False for k in task_ids}
        # clear lock file
        if rank == 0 and os.path.exists(self.lock_fname):
            os.remove(self.lock_fname)
            print("cleared lock file")
        d = OrderedDict()
        for task_id in task_ids:
            d[task_id] = np.zeros(num_completions_per_task, dtype=bool)
        # save d to file
        if rank == 0:
            # save dictionary d as pickle
            pickle.dump(d, open(fname, "wb"))
            # save dictionary d as npy
            print(f"Saved to {fname}")
        self.d = d

    def check_lock(self, rank, const=0.0123):
        time.sleep(const * rank)  # prefer lower rank
        while os.path.isfile(self.lock_fname):
            time.sleep(const)
        if self.verbose:
            print(f"rank {rank} lock.")
        # there can be accidents -- placing another check
        if os.path.isfile(self.lock_fname):
            self.check_lock(rank)
        Path(self.lock_fname).touch()
        return True

    def release_lock(self):
        try:
            if os.path.isfile(self.lock_fname):
                os.remove(self.lock_fname)
                return True
        except Exception:
            return False

    def mark_job_scheduled(self, taskid, completion_ids, rank):
        self.d[taskid][completion_ids] = True
        pickle.dump(self.d, open(self.fname, "wb"))
        np.save(self.fname, self.d)
        self.release_lock()

    def __iter__(self):
        for task_id in self.task_ids:
            while not self.task_done[task_id]:
                # wait until check lock returns
                self.check_lock(self.rank)
                # if the file is damaged, we recover from the previous
                # get the last job that are still marked as False
                try:
                    self.d = pickle.load(open(self.fname, "rb"))
                except Exception:
                    pass
                if not np.all(self.d[task_id]):
                    start_id = np.argmin(self.d[task_id])
                    end_id_exclusive = min(
                        start_id + self.batch_size, self.num_completions_per_task
                    )
                    completion_ids = range(start_id, end_id_exclusive)
                    self.mark_job_scheduled(
                        task_id, completion_ids=completion_ids, rank=self.rank
                    )
                    yield task_id, list(completion_ids)
                else:
                    self.task_done[task_id] = True
                    self.release_lock()
                    break

    def __len__(self):
        num_splits_per_task = math.ceil(
            self.num_completions_per_task / (1.0 * self.batch_size * self.num_groups)
        )
        num_subtasks_per_group = num_splits_per_task * len(self.task_ids)
        return num_subtasks_per_group

    def get_tqdm_bar(self):
        return tqdm(
            iter(self),
            total=len(self),
            unit_scale=1
            / (
                math.ceil(
                    self.num_completions_per_task
                    / (1.0 * self.batch_size * self.num_groups)
                )
            ),
        )