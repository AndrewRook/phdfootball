from prefect.core.task import Task
from prefect.engine.state import State, Success
from prefect.engine.result import Result
from prefect.core.edge import Edge
import typing

def disk_state_handler(task: Task, old_state: State, new_state: State) -> State:
    if task.task.checkpoint and old_state.is_pending() and new_state.is_running():
        input_mapping = _create_input_mapping(task.upstream_states)
        try:
            data = task.task.result_handler.read(input_mapping=input_mapping)
        except FileNotFoundError:
            return new_state
        result = Result(value=data, result_handler=task.task.result_handler)
        state = Success(result=result, message="Task loaded from disk.")
        # task.task.checkpoint = False # Doesn't work with mapped tasks :(
        return state


    if task.task.checkpoint and old_state.is_running() and new_state.is_successful():
        input_mapping = _create_input_mapping(task.upstream_states)
        task.task.result_handler.write(new_state.result, input_mapping=input_mapping)

    #breakpoint()
    return new_state


def _create_input_mapping(upstream_states: typing.Dict[Edge, State]) -> typing.Dict[str, typing.Any]:
    mapping = {}
    for edge, state in upstream_states.items():
        input_variable_name = edge.key
        input_task_result = state.result
        mapping[input_variable_name] = input_task_result
    return mapping