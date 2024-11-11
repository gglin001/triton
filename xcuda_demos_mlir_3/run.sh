################################################################################

rm -rf _demos/.triton/cache
rm -rf _demos/.triton/dump

python xcuda_demos_mlir_3/graph.py

################################################################################

mkdir -p _demos/tx.tmp

MLIR_ENABLE_DUMP=1 \
  MLIR_ENABLE_DUMP_DIR="_demos/xcuda_demos_mlir_3" \
  python xcuda_demos_mlir_3/graph.py 2>&1 0>&1 | tee _demos/tx.tmp/graph.log

################################################################################
