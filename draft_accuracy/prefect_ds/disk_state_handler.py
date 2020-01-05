from prefect.core.task import Task
from prefect.engine.state import State, Success
from prefect.engine.result import Result


def disk_state_handler(task: Task, old_state: State, new_state: State) -> State:
    if task.task.checkpoint and old_state.is_pending() and new_state.is_running():
        try:
            data = task.task.result_handler.read()
        except FileNotFoundError:
            return new_state
        result = Result(value=data, result_handler=task.task.result_handler)
        state = Success(result=result, message="Task loaded from disk.")
        task.task.checkpoint = False
        return state


    if task.task.checkpoint and old_state.is_running() and new_state.is_successful():
        task.task.result_handler.write(new_state.result)

    #breakpoint()
    return new_state