from __future__ import print_function
import ctypes
import pickle

from ctypes import CFUNCTYPE, c_double, c_int
import time

import llvmlite.binding as llvm

import llvmlite.binding
# All these initializations are required for code generation!
llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()  # yes, even this one

# load stdc++
# llvm.load_library_permanently("libstdc++abi.so")

# llvm.load_library_permanently("/home/yy/library/clang/lib/libc++.so.1.0")
llvm.load_library_permanently("libstdc++.so.6")
llvm.load_library_permanently("libc.so.6")

# the following ir compiled from this source by clang++ -S -emit-llvm -c test.cc -O3 -o test.ll
"""
#include <iostream>

int test(){
    return 0;
}
"""
llvm_ir = r"""
; ModuleID = '/tmp/tmp2iagli1g.cc'
source_filename = "/tmp/tmp2iagli1g.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"class.std::ios_base::Init" = type { i8 }

@_ZStL8__ioinit = internal global %"class.std::ios_base::Init" zeroinitializer, align 1
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_tmp2iagli1g.cc, i8* null }]

declare dso_local void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"*) unnamed_addr #0

; Function Attrs: nounwind
declare dso_local void @_ZNSt8ios_base4InitD1Ev(%"class.std::ios_base::Init"*) unnamed_addr #1

; Function Attrs: nounwind uwtable
define internal void @__dtor__ZStL8__ioinit() #2 section ".text.startup" {
  tail call void @_ZNSt8ios_base4InitD1Ev(%"class.std::ios_base::Init"* nonnull @_ZStL8__ioinit)
  ret void
}

; Function Attrs: nounwind
declare dso_local i32 @atexit(void ()*) local_unnamed_addr #3

; Function Attrs: norecurse nounwind readnone uwtable
define dso_local void @_ZN23__pccm_inline_namespace22__pccm_inline_functionEv() local_unnamed_addr #4 {
  ret void
}

; Function Attrs: uwtable
define internal void @_GLOBAL__sub_I_tmp2iagli1g.cc() #5 section ".text.startup" {
  tail call void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"* nonnull @_ZStL8__ioinit)
  %1 = tail call i32 @atexit(void ()* nonnull @__dtor__ZStL8__ioinit) #3
  ret void
}

attributes #0 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }
attributes #4 = { norecurse nounwind readnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 11.0.0 (https://github.com/llvm/llvm-project.git 0160ad802e899c2922bc9b29564080c22eb0908c)"}


"""

# with open("/home/yy/Projects/cumm/build/dev_llvm_ir.ll", "r") as f:
#   llvm_ir = f.read()

def create_execution_engine():
    """
    Create an ExecutionEngine suitable for JIT code generation on
    the host CPU.  The engine is reusable for an arbitrary number of
    modules.
    """
    # Create a target machine representing the host
    target = llvm.Target.from_default_triple()
    target_machine = target.create_target_machine()
    # And an execution engine with an empty backing module
    backing_mod = llvm.parse_assembly("")
    engine = llvm.create_mcjit_compiler(backing_mod, target_machine)
    return engine

def compile_ir(engine: llvm.ExecutionEngine, llvm_ir):
    """
    Compile the LLVM IR string with the given engine.
    The compiled module object is returned.
    """
    # Create a LLVM module object from the IR
    mod = llvm.parse_assembly(llvm_ir)
    mod.verify()
    # Now add the module and make sure it is ready for execution
    engine.add_module(mod)
    engine.finalize_object()
    print(1)
    engine.run_static_constructors()
    print(2)
    return mod


def main():
    engine = create_execution_engine()

    mod = compile_ir(engine, llvm_ir)

    # Look up the function pointer (a Python int)
    func_ptr = engine.get_function_address(
        "_ZN23__pccm_inline_namespace22__pccm_inline_functionEv")
    print(func_ptr)
    assert func_ptr != 0
    # Run the function via ctypes
    cfunc = CFUNCTYPE(c_int)(func_ptr)
    res = cfunc()
    print("fpadd(...) =", res)
    print("1")
    engine.run_static_destructors()
    # mod.close()
    # engine.close()
    # del mod 
    # del engine
    print(222)
    # time.sleep(5)


# if __name__ == "__main__":
main()
