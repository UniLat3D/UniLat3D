import sys
import traceback
from prompt_toolkit import prompt
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings

from cumm.common import TensorViewArrayLinalg, TensorViewNVRTCHashKernel, TensorViewNVRTC, TensorViewCPU
from cumm.inliner import NVRTCInlineBuilder
from prompt_toolkit.history import InMemoryHistory
import pccm
from typing import List, Optional, Type
from cumm import tensorview as tv
from pygments.lexers.c_cpp import CppLexer
from prompt_toolkit.lexers import PygmentsLexer


class InlineCUDATerminal:

    def __init__(self,
                 deps: Optional[List[Type[pccm.Class]]] = None,
                 param_deps: Optional[List[pccm.ParameterizedClass]] = None,
                 is_cpu: bool = False) -> None:
        self._history = []
        self._global_count = 1
        self._enter_multi_line = False
        self._history_index = -1
        self._is_cpu = is_cpu
        if not is_cpu:
            tv.zeros([1], device=0)  # init driver
        if deps is None:
            deps = []
        if not is_cpu:
            deps.extend([
                TensorViewArrayLinalg, TensorViewNVRTCHashKernel,
                TensorViewNVRTC
            ])
        else:
            deps.extend([TensorViewCPU, TensorViewArrayLinalg])
        self._inliner = NVRTCInlineBuilder(deps,
                                           param_deps=param_deps,
                                           reload_when_code_change=True,
                                           std="c++17")

        self.kb = KeyBindings()
        self.kb.add('enter')(self._on_enter)

    def _on_enter(self, event):
        if not self._enter_multi_line:
            cur_text_strip = event.current_buffer.text.strip()
            if cur_text_strip.endswith('\\') or cur_text_strip.endswith('{'):
                self._enter_multi_line = True
                event.current_buffer.insert_text('\n')
                return
            else:
                self._global_count += 1
                event.current_buffer.validate_and_handle()
        else:
            lines = event.current_buffer.text.split("\n")
            if lines[-1].strip() == "":
                self._enter_multi_line = False
                event.current_buffer.validate_and_handle()
                self._global_count += 1
                return
            else:
                event.current_buffer.insert_text('\n')
                return

    def prompt_continuation(self, width, line_number, wrap_count):
        """
        The continuation: display line numbers and '->' before soft wraps.

        Notice that we can return any kind of formatted text from here.

        The prompt continuation doesn't have to be the same width as the prompt
        which is displayed before the first line, but in this example we choose to
        align them. The `width` input that we receive here represents the width of
        the prompt.
        """
        # print(text.encode("utf-8"))
        if wrap_count > 0:
            # return " " * (width - 3) + "-> "
            return HTML(f'<style fg="green">   ...: </style>')

        else:
            # text = ("- %i - " % (line_number + 1)).rjust(width)
            return HTML(f'<style fg="green">   ...: </style>')

    def prompt_main(self):
        # print("Press [Meta+Enter] or [Esc] followed by [Enter] to accept input.")
        history = InMemoryHistory()
        while True:
            first_html = HTML(
                f'<style fg="green">In [<strong>{self._global_count}</strong>]: </style>'
            )
            answer = prompt(first_html,
                            multiline=True,
                            key_bindings=self.kb,
                            prompt_continuation=self.prompt_continuation,
                            history=history,
                            lexer=PygmentsLexer(CppLexer))
            self._history.append(answer)
            if answer.strip() == "exit":
                return
            if answer.strip() == "":
                continue
            sys.stdout.flush()

            try:
                if self._is_cpu:
                    answer = answer.rstrip()
                    if not answer.endswith(";"):
                        # trait it with expression, try to use tv::printf2
                        answer = f"tv::ssprint({answer});"
                    self._inliner.cpu_kernel_raw("prompt_main", answer)
                else:
                    answer = answer.rstrip()
                    if not answer.endswith(";"):
                        # trait it with expression, try to use tv::printf2
                        answer = f"tv::printf2({answer});"
                    self._inliner.kernel_1d("prompt_main", 1, 0, answer)

            except:
                traceback.print_exc()
            if not self._is_cpu:
                # inliner run kernels in stream. we need to sync it.
                ctx = tv.Context()
                ctx.set_cuda_stream(0)
                ctx.synchronize_stream()
