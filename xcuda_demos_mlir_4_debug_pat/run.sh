###############################################################################

# bash xcuda_demos_mlir_4_debug_pat/run.sh

###############################################################################

ENABLE_PATTERNS=""
ENABLE_PATTERNS+="\""
ENABLE_PATTERNS+="(anonymous namespace)::FuncOpConversion"
ENABLE_PATTERNS+=",(anonymous namespace)::ReturnOpConversion"
#
# ENABLE_PATTERNS+=",(anonymous namespace)::WarpGroupDotOpConversion"
# ENABLE_PATTERNS+=",(anonymous namespace)::IndexCastOpLowering"
#
ENABLE_PATTERNS+="\""

DISABLE_PATTERNS=""
DISABLE_PATTERNS+="\""
DISABLE_PATTERNS+="(anonymous namespace)::WarpGroupDotOpConversion"
DISABLE_PATTERNS+="\""

CONVERT_TRITON_GPU_TO_LLVM="compute-capability=90 enable-patterns=$ENABLE_PATTERNS disable-patterns=$DISABLE_PATTERNS"
# CONVERT_TRITON_GPU_TO_LLVM="compute-capability=90 enable-patterns=$ENABLE_PATTERNS"
# CONVERT_TRITON_GPU_TO_LLVM="compute-capability=90 disable-patterns=$DISABLE_PATTERNS"

args=(
  -split-input-file
  -mlir-print-stacktrace-on-diagnostic
  #
  # --allocate-shared-memory
  #
  # --convert-triton-gpu-to-llvm
  --convert-triton-gpu-to-llvm="$CONVERT_TRITON_GPU_TO_LLVM"
  #
  # -debug-only='dialect-conversion,greedy-rewriter,pattern-application'
  # -debug-only='dialect-conversion'
  -debug-only='pattern-application'
  #
  xcuda_demos_mlir_4_debug_pat/graph.mlir
  -o xcuda_demos_mlir_4_debug_pat/graph.mlir.opt.mlir
)
set -x
triton-opt "${args[@]}" 2>&1 0>&1 | tee xcuda_demos_mlir_4_debug_pat/run.sh.log

###############################################################################
