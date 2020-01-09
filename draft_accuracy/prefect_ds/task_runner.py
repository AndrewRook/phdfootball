from prefect.core import Edge
from prefect.engine.state import State
from prefect.engine.task_runner import TaskRunner
from typing import Dict, Any


class DSTaskRunner(TaskRunner):
    def run(
        self,
        state: State = None,
        upstream_states: Dict[Edge, State] = None,
        context: Dict[str, Any] = None,
        executor: "prefect.engine.executors.Executor" = None,
    ) -> State:
        self.upstream_states = upstream_states
        return super().run(state=state, upstream_states=upstream_states, context=context, executor=executor)
