# Copyright 2024 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import contextvars
from typing import Any, Optional

DEBUG_CONTEXT_VAR = contextvars.ContextVar("CudasimDebug")


class DebugContext:
    def __init__(self, enable: bool = True, debug_tx: int = 0):
        self.enable = enable
        self.debug_tx = debug_tx


def get_debug_context() -> Optional[DebugContext]:
    return DEBUG_CONTEXT_VAR.get(None)


def enable_debug():
    ctx = get_debug_context()
    if ctx is None:
        return False
    return ctx.enable


def debug_tx():
    ctx = get_debug_context()
    if ctx is None:
        return -1
    if not ctx.enable:
        return -1
    return ctx.debug_tx


@contextlib.contextmanager
def enter_debug_context(enable: bool = True, debug_tx: int = 0):
    ctx = DebugContext(enable, debug_tx)
    token = DEBUG_CONTEXT_VAR.set(ctx)
    yield ctx
    DEBUG_CONTEXT_VAR.reset(token)


def debug_print(*args: Any):
    ctx = get_debug_context()
    if ctx is None:
        return print(*args)
    else:
        if ctx.enable:
            return print(*args)
    return
