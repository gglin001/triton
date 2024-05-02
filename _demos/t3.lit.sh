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
