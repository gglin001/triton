#ifndef TRITONGPU_CONVERSION_TRITONNVIDIAGPUTOLLVM_PASSES_H
#define TRITONGPU_CONVERSION_TRITONNVIDIAGPUTOLLVM_PASSES_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

#define GEN_PASS_DECL
#include "nvidia/include/TritonNVIDIAGPUToLLVM/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonGPUToLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGPUToLLVMPass(int32_t computeCapability);
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGPUToLLVMPass(int32_t computeCapability, int32_t ptxVersion);

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGPUToLLVMPass(int32_t computeCapability,
                                 ArrayRef<std::string> disabledPatterns,
                                 ArrayRef<std::string> enabledPatterns);
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGPUToLLVMPass(int32_t computeCapability, int32_t ptxVersion,
                                 ArrayRef<std::string> disabledPatterns,
                                 ArrayRef<std::string> enabledPatterns);

#define GEN_PASS_REGISTRATION
#include "nvidia/include/TritonNVIDIAGPUToLLVM/Passes.h.inc"

} // namespace triton

} // namespace mlir

#endif
