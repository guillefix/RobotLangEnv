

def distribute_tasks(tasks, rank, size):
    num_tasks = len(tasks)
    num_tasks_per_job = num_tasks//size
    tasks_ids = list(range(rank*num_tasks_per_job,(rank+1)*num_tasks_per_job))
    if rank < num_tasks%size:
        tasks_ids.append(size*num_tasks_per_job+rank)
    tasks = [tasks[i] for i in tasks_ids]
    return tasks
