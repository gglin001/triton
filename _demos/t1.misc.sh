triton-opt --help >_demos/triton-opt.help.log
triton-llvm-opt --help >_demos/triton-llvm-opt.help.log
triton-shared-opt --help >_demos/triton-shared-opt.help.log

lit --help >_demos/lit.help.log
FileCheck --help >_demos/FileCheck.help.log
FileCheck --dump-input=help

###############################################################################
