//===- TestLinalgDependantReductionFusion.cpp - Test reduction fusion -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test pass for fusing a dependent `R1 -> E -> R2` reduction chain into `R1`'s
// already-tiled loop. The entry point for the transformation is
// `transform.structured.fuse_dependant_reduction_ops`, which fuses only the
// chain named by its handles; this pass drives the same patterns over a whole
// function with an unconstrained control function, so the legality conditions
// and the generated IR can be tested directly.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

struct TestLinalgDependantReductionFusion
    : public PassWrapper<TestLinalgDependantReductionFusion,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestLinalgDependantReductionFusion)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
  }
  StringRef getArgument() const final {
    return "test-linalg-dependant-reduction-fusion";
  }
  StringRef getDescription() const final {
    return "Test fusion of a dependent R1 -> E -> R2 reduction chain into the "
           "already-tiled producer reduction loop";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // Unconstrained control function: allow fusing through any operand whose
    // producer is a defined operation. The transform op supplies a control
    // function that narrows this to a single explicitly named chain.
    linalg::ControlFusionFn controlFn = [](OpOperand *fusedOperand) {
      return fusedOperand->get().getDefiningOp() != nullptr;
    };
    linalg::populateDependantReductionFusionPatterns(patterns, controlFn);

    GreedyRewriteConfig config;
    config.enableFolding(false);
    // For chains R1 -> R2 -> R3, match the upstream consumer (R2) before the
    // downstream one (R3) so (R1, R2) fuses first and the resulting op then
    // fuses with R3 on the next worklist pass.
    config.setUseTopDownTraversal(true);
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestLinalgDependantReductionFusion() {
  PassRegistration<TestLinalgDependantReductionFusion>();
}
} // namespace test
} // namespace mlir
