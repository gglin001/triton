###############################################################################

args=(
  build/test
  --show-tests
  --debug
)
lit "${args[@]}"

###############################################################################

args=(
  build/test
  # --no-execute
  # --debug
  -v -a
  --filter "Conversion/amd/fp_to_fp.mlir"
)
lit "${args[@]}"
# lit "${args[@]}" | tee _demos/lit.run.dirty.mlir

###############################################################################

# alias FileCheck="FileCheck --vv --dump-input=always --color=1"
# which FileCheck

export FILECHECK_OPTS="--vv --dump-input=always --color=1"

###############################################################################

triton-opt test/Conversion/amd/fp_to_fp.mlir --split-input-file --convert-triton-amdgpu-to-llvm=arch=gfx942 |
  FileCheck test/Conversion/amd/fp_to_fp.mlir
