from cumm.inliner.prompt import InlineCUDATerminal

def cpu():
    print("Cumm Clang++ Prompt. each block is a standalone program."
          " use 'exit' to exit. you must install clang to system first and "
          "llvmlite/Pygments.")
    handler = InlineCUDATerminal(is_cpu=True)
    handler.prompt_main()

if __name__ == "__main__":
    cpu()
