import contextlib
from typing import Dict, List, Optional, Tuple
from cumm import tensorview as tv

import contextvars


class PerfContext:
    def __init__(self) -> None:
        self._ns_stack: List[str] = []
        self._measures: Dict[Tuple[str, ...], List[Tuple[tv.CUDAEvent,
                                                         tv.CUDAEvent]]] = {}
        self._print_pair: List[Tuple[str, float]] = []
        self.perf_result: Dict[Tuple[str, ...], List[float]] = {}
        self._enable_controlled_by_root: Optional[bool] = None


PERF_CONTEXT: contextvars.ContextVar[
    Optional[PerfContext]] = contextvars.ContextVar("perf_context",
                                                    default=None)


@contextlib.contextmanager
def __enter_perf_conetxt(perf_ctx: PerfContext):
    token = PERF_CONTEXT.set(perf_ctx)
    try:
        yield perf_ctx
    finally:
        PERF_CONTEXT.reset(token)


@contextlib.contextmanager
def perf_context(name: str,
                 *,
                 stream: int = 0,
                 enable: bool = True,
                 print_result: bool = True,
                 control_child_enable: bool = False):
    ctx = PERF_CONTEXT.get()
    enter_null = contextlib.nullcontext()
    is_root = False
    root_key = None
    if ctx is None:
        ctx = PerfContext()
        if control_child_enable:
            ctx._enable_controlled_by_root = enable
        is_root = True
        root_key = (name, )
        enter_null = __enter_perf_conetxt(ctx)
    if ctx._enable_controlled_by_root is not None:
        enable = ctx._enable_controlled_by_root
    if not enable:
        yield None
        return
    ctx._ns_stack.append(name)
    root_time = 1
    try:
        with enter_null:
            ev_start = tv.CUDAEvent("")
            ev_stop = tv.CUDAEvent("")
            ev_start.record(stream)
            yield ctx
            ev_stop.record(stream)
            key = tuple(ctx._ns_stack)
            if key not in ctx._measures:
                ctx._measures[key] = []
            ctx._measures[tuple(ctx._ns_stack)].append((ev_start, ev_stop))

    finally:
        ctx._ns_stack.pop()
    if is_root:
        all_times: Dict[Tuple[str, ...], List[float]] = {}
        for key, data in ctx._measures.items():
            for pair in data:
                pair[0].sync()
                pair[1].sync()
            times = [tv.CUDAEvent.duration(x[0], x[1]) for x in data]
            all_times[key] = times
            if key == root_key:
                root_time = times[0]
        ctx.perf_result = all_times
        ctx._measures.clear()
        if print_result:
            for key, data in all_times.items():
                time = sum(data, 0)
                if len(key) > 1:
                    print(
                        f"[{key[-1]}@{len(data)}]({(time / root_time) * 100:.3f}%): {time:.4}"
                    )
                else:
                    print(f"[{key[-1]}@{len(data)}]: {time:.4}")
