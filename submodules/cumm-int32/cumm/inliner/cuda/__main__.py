from cumm.inliner.prompt import InlineCUDATerminal

def cuda():
    print("Cumm NVRTC Kernel Prompt. each block is a standalone kernel. "
          "use 'exit' to exit. All headers in tensorview/core are available. "
          "The kernel run with one thread and one block.")
    handler = InlineCUDATerminal()
    handler.prompt_main()

if __name__ == "__main__":
    cuda()