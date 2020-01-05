from prefect.core.task import Task
from prefect.engine.state import State

def disk_state_handler(task: Task, old_state: State, new_state: State) -> State:
    if task.task.checkpoint and old_state.is_pending() and new_state.is_running():
        breakpoint()
