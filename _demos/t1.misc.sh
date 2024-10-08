###############################################################################

mkdir -p _demos/_log

triton-opt --help >_demos/_log/triton-opt.help.log
triton-llvm-opt --help >_demos/_log/triton-llvm-opt.help.log
triton-reduce --help >_demos/_log/triton-reduce.help.log
# triton-shared-opt --help >_demos/_log/triton-shared-opt.help.log

lit --help >_demos/_log/lit.help.log
FileCheck --help >_demos/_log/FileCheck.help.log
FileCheck --dump-input=help

###############################################################################
