import math
import os

TASK_ID = os.environ.get('SGE_TASK_ID')
LAST_TASK_ID = os.environ.get('SGE_TASK_LAST')
TASK_STEPSIZE = os.environ.get('SGE_TASK_STEPSIZE')
ISTITAN = os.environ.get('ISTITAN')

def is_cluster():
    return not (ISTITAN is None)

def get_num_tasks():
    if not TASK_ID:
        return 1
    return int(math.ceil(float(LAST_TASK_ID)/float(TASK_STEPSIZE)))

def get_task_id():
    try:
        return int(TASK_ID)
    except TypeError:
        return 1

def crange(in_range):
    """
    split the range up equally amongst the tasks (tasks are alwayssequential e.g. 1-80:1)
    """
    n_tasks = get_num_tasks()
    if n_tasks == 1:
        return in_range

    if n_tasks < len(in_range):
        split = math.floor(float(len(in_range))/float(n_tasks))
        dist = [split for i in range(n_tasks)]
        remainder = len(in_range) - (split*n_tasks)
        # distribute this remainder across the first tasks
        i = 0
        while remainder > 0:
            dist[i] += 1
            remainder -= 1
            i += 1
        start_i = int(sum(dist[0:int(TASK_ID)-1]))
        end_i = int(start_i + dist[int(TASK_ID)-1])
        if int(TASK_ID) == n_tasks:
            out_range = in_range[start_i:]
        else:
            out_range = in_range[start_i:end_i]
        print 'Task %d of %d, split is %d of %d' % (int(TASK_ID), n_tasks, len(out_range), len(in_range))
        return out_range
    else:
        # if less range than tasks then just process one on each task and quit the rest
        if int(TASK_ID) <= len(in_range):
            return [in_range[int(TASK_ID)-1]]
        else:
            print 'Nothing to process'
            return []
