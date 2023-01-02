import json
import os
import gzip
import sys
import subprocess

def read_problems(eval_file):
    return {str(task["task_id"]): task for task in stream_jsonl(eval_file)}

def stream_jsonl(filename):
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, "rt") as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)

def run_passatk_eval(
    problem_file,
    language,
    output_dir,
    num_samples_per_example,
    override_previous_results,
):
    # aggregate all generations in a jsonl file
    output_samples_file = os.path.join(output_dir, "samples.jsonl")
    file_exists = os.path.isfile(output_samples_file)
    problems = read_problems(problem_file)
    if file_exists:
        samples = read_problems(output_samples_file)
    else:
        samples = []

    if override_previous_results or not (file_exists and len(problems) == len(samples)):
        with open(output_samples_file, "w") as fw:
            for task_id in problems:
                task_idx = int(task_id.split("/")[1])
                for completion_idx in range(num_samples_per_example):
                    _fname = os.path.join(
                        output_dir,
                        "output",
                        f"taskid-{task_idx}-gen{completion_idx}.json",
                    )
                    prediction = json.load(open(_fname, "r", encoding="utf8"))
                    fw.write(json.dumps(prediction) + "\n")

    # evaluate with pass@k metrics
    local_evaluate_functional_correctness(
        output_samples_file, problem_file
    )


def local_evaluate_functional_correctness(
    output_samples_file, problem_file, aggregate_file=None
):
    """
    Use subprocess to execute so that the num processes are not bottlenecked
    by pytorch. If we call evaluate_functional_correctness module directly,
    the number of processes can be limited due to not mamy workers being available
    which results in very slow execution.
    """
    print(f"Evaluating from {output_samples_file}")
    command = f"evaluate_functional_correctness --sample_file {output_samples_file} --problem_file {problem_file} --n_workers {os.cpu_count()}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    for c in iter(lambda: process.stdout.read(1), b""):
        sys.stdout.buffer.write(c)
    if aggregate_file is not None:
        s = open(output_samples_file + "_passatk.txt", "r").read()
        aggregate_file.write(s + "\n")