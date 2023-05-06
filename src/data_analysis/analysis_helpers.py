from pandas import np


def get_run_ids(runs):
    for row, data in runs.iterrows():
        if data["status"] == "RUNNING":
            continue
        run_id = data["run_id"]
    run_ids = runs.run_id
    return run_ids

def get_out_dirs(run_ids):
    out_dir = '/nfs/data3/obermeier/dmbal/experiments/' + run_ids.array[0]
    out_dirs = np.array(out_dir)
    i=1
    while i < len(run_ids.array):
        out_dir = '/nfs/data3/obermeier/dmbal/experiments/' + run_ids.array[i]
        out_dirs = np.append(out_dirs, out_dir)
        i+=1
    return out_dirs

def get_labeledSamples_avgAcc(run_ids, tracking):
    all_accs = []
    for run_id in run_ids:
        all_accs.append(np.array([m.value for m in tracking.get_metric_history(run_id, "acc")]))
        samples_labeled = np.array([m.value for m in tracking.get_metric_history(run_id, "samples_labeled")])
    all_accs = np.array(all_accs)
    avg_accs = np.mean(all_accs, axis=0)
    return samples_labeled, avg_accs

def get_labeledSamples_avgAcc_withBounds(run_ids, tracking):
    all_accs = []
    for run_id in run_ids:
        all_accs.append(np.array([m.value for m in tracking.get_metric_history(run_id, "acc")]))
        samples_labeled = np.array([m.value for m in tracking.get_metric_history(run_id, "samples_labeled")])
    all_accs = np.array(all_accs)
    avg_accs = np.mean(all_accs, axis=0)
    std_accs = np.std(all_accs, axis=0)
    lower_bound = avg_accs - std_accs
    upper_bound = avg_accs + std_accs
    return samples_labeled, avg_accs, lower_bound, upper_bound