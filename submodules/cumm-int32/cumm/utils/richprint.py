from typing import List, Any, Optional, Set, Tuple
from typing import IO


try:
    import rich 
    HAS_RICH_PRINT = True
    rprint = rich.print 
except ImportError:
    HAS_RICH_PRINT = False
    rprint = print

def print_rich_cpp(code: str, highlights: Set[int], line_range: Optional[Tuple[Optional[int], Optional[int]]] = None):
    if HAS_RICH_PRINT:
        from rich.syntax import Syntax
        syntax = Syntax(code, "cpp", theme="monokai", line_numbers=True, highlight_lines=highlights, word_wrap=True, line_range=line_range)
        return rprint(syntax)
    else:
        lines = code.split("\n")
        for lineno in highlights:
            lines[lineno - 1] = f"!!!!!! {lines[lineno-1]}"
        if line_range is not None:
            start_lineno, end_lineno = line_range
            if start_lineno is None:
                start_lineno = 1
            if end_lineno is None:
                end_lineno = len(lines)
            start_lineno = max(start_lineno, 1)
            end_lineno = min(end_lineno, len(lines))
            lines = lines[start_lineno-1:end_lineno]
        modified_lines = "\n".join(lines)
        return rprint(modified_lines)