//===- DependantReductionFusion.cpp - Fuse dependent reductions -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file fuses a chain of dependent reductions sharing a reduction dimension
// into a single online (one-pass) loop. The canonical case is two-pass softmax
// (max, then sum-of-exp) becoming the online form.
//
// The chain is `R1 -> E -> R2`:
//   R1  a reduction already tiled into a `__reduction_loop__` `scf.for`,
//   E   an all-parallel elementwise op reading R1's result plus the data inputs
//       it shares with R1,
//   R2  a reduction of E's result, possibly alongside other operands (e.g. a
//       GEMM-like contraction), all reduced along R1's reduction axis.
// `E` and `R2` are cloned into R1's loop and re-sliced to the reduction tile,
// then R2's running accumulator is rescaled per tile by a correction factor
// derived from `E` (see `calculateCorrectionFactor`).
//
// Exposed only as a pattern; the entry point is
// `transform.structured.fuse_dependant_reduction_ops`, whose control function
// restricts fusion to one named chain. There is no standalone pass, as fusing
// every chain in a function is not a useful default. The
// `test-linalg-dependant-reduction-fusion` pass drives the pattern
// unconstrained for testing.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "dependant-reduction-fusion"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

using namespace mlir;
using namespace mlir::linalg;

namespace {

/// Unit attribute marking an `scf.for` as a tiled reduction loop. A plain
/// `scf.for` carries no iterator-type metadata, so this marker is the only way
/// to recognise an already-tiled producer reduction `R1`.
static constexpr StringLiteral kReductionLoopAttrName = "__reduction_loop__";

/// Resolve `val` through any chain of `tensor.extract_slice` ops to the source
/// tensor. The inner `R1` reads slices of the real inputs, so comparing its
/// inputs against `E`'s has to look through them.
static Value resolveSliceSource(Value val) {
  while (auto slice = val.getDefiningOp<tensor::ExtractSliceOp>())
    val = slice.getSource();
  return val;
}

/// Return true if `val` is statically known to be zero -- either a constant
/// zero scalar/splat, or chained through a `linalg.fill` / `linalg.copy` of a
/// zero value. Mirrors the helper in FoldAddIntoDest.cpp.
static bool isDefinedAsZero(Value val) {
  if (!val)
    return false;
  if (isZeroIntegerOrFloat(val))
    return true;
  Operation *defOp = val.getDefiningOp();
  if (!defOp)
    return false;
  return TypeSwitch<Operation *, bool>(defOp)
      .Case<linalg::FillOp, linalg::CopyOp>([&](auto op) {
        return op.getInputs().size() == 1 && isDefinedAsZero(op.getInputs()[0]);
      })
      .Default([&](auto) { return false; });
}

/// The `E` input operands consuming an R1 result, paired with the index of the
/// result each one consumes. Legality needs both: the operands to check their
/// indexing maps, the index to recover the inner reduction producing them.
struct R1AsElementwiseInputInfo {
  SmallVector<OpOperand *> operands;
  SmallVector<unsigned> r1ResultIdx;
};

/// Collect the `e` DPS inputs reading a result of the `__reduction_loop__` loop
/// `r1`, each paired with the loop-result index it reads.
static R1AsElementwiseInputInfo
collectR1AsElementwiseInputs(Operation *r1, linalg::GenericOp e) {
  R1AsElementwiseInputInfo info;
  for (OpOperand *in : e.getDpsInputOperands()) {
    for (unsigned i = 0; i < r1->getNumResults(); ++i) {
      if (in->get() == r1->getResult(i)) {
        info.operands.push_back(in);
        info.r1ResultIdx.push_back(i);
        break;
      }
    }
  }
  return info;
}

/// The reduction `linalg.generic`s in the loop body, in program order. A loop
/// may carry several running accumulators (e.g. a fused softmax `(max, sum)`
/// loop), each produced by its own inner reduction.
static SmallVector<linalg::GenericOp>
collectInnerReductionGenerics(scf::ForOp loop) {
  SmallVector<linalg::GenericOp> result;
  for (Operation &op : loop.getBody()->without_terminator()) {
    auto generic = dyn_cast<linalg::GenericOp>(&op);
    if (!generic || generic.getNumReductionLoops() == 0)
      continue;
    result.push_back(generic);
  }
  return result;
}

/// Map each loop result to the inner reduction producing its tile, indexed by
/// result number; null for results not yielded from a reduction (e.g. a
/// passthrough). Accumulators are yielded as `tensor.insert_slice %red into
/// %arg`, so each `scf.yield` operand is matched against that shape.
static SmallVector<linalg::GenericOp>
mapLoopResultsToInnerReductions(scf::ForOp loop) {
  SmallVector<linalg::GenericOp> resultToInner(loop.getNumResults(), nullptr);
  auto yieldOp = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  for (auto [idx, yielded] : llvm::enumerate(yieldOp.getOperands())) {
    auto insertSlice = yielded.getDefiningOp<tensor::InsertSliceOp>();
    if (!insertSlice)
      continue;
    auto generic = insertSlice.getSource().getDefiningOp<linalg::GenericOp>();
    if (generic && generic.getNumReductionLoops() != 0)
      resultToInner[idx] = generic;
  }
  return resultToInner;
}

/// Check that one inner reduction `r1` of the loop can be fused against `e`:
///   - `r1` reduces exactly one dim, its innermost;
///   - every `r1` input also appears as an `e` input, except results of a
///   sibling
///     inner reduction (`innerResults`), which are broadcast accumulators;
///   - the shared inputs' maps agree under a dim mapping phi from `r1`'s loops
///   to
///     `e`'s, total over `r1`'s loops and taking `r1`'s reduction dim to
///     `eTiledDim` (the `e` dim carrying R2's reduction axis).
/// The comparison is against `E` rather than `R2` because `E` is what shares
/// `R1`'s data inputs; `R2` need not share any.
static LogicalResult checkInnerReductionAgainstElementwise(
    linalg::GenericOp r1, linalg::GenericOp e, unsigned eTiledDim,
    const llvm::SmallDenseSet<Value, 4> &innerResults) {
  SmallVector<unsigned> r1RedDims;
  r1.getReductionDims(r1RedDims);
  if (r1RedDims.size() != 1) {
    LLVM_DEBUG(
        DBGS() << "checkInnerReductionAgainstElementwise: failed -- inner "
                  "R1 does not have exactly one reduction iterator ("
               << r1RedDims.size() << ").\n");
    return failure();
  }
  if (r1RedDims.front() != r1.getNumLoops() - 1) {
    LLVM_DEBUG(DBGS() << "checkInnerReductionAgainstElementwise: failed -- "
                         "reduction iterator is not the innermost loop in "
                         "inner R1.\n");
    return failure();
  }

  // Every input of R1 must also appear as an input of E (besides R1's result
  // and any sibling reduction's result), and the two ops' indexing maps for
  // each shared input must agree up to a consistent injective mapping phi from
  // R1's loop dims to E's loop dims.
  llvm::SmallDenseMap<unsigned, unsigned, 4> phi;
  auto tryAddMapping = [&](unsigned r1Dim, unsigned eDim) -> bool {
    auto it = phi.find(r1Dim);
    if (it == phi.end()) {
      phi[r1Dim] = eDim;
      return true;
    }
    return it->second == eDim;
  };
  for (OpOperand *in1 : r1.getDpsInputOperands()) {
    // The inner R1 reads a `tensor.extract_slice` of the real input tensor;
    // resolve through the slice to compare against E's (untiled) inputs.
    Value in1Source = resolveSliceSource(in1->get());
    // Inputs that are results of a sibling inner reduction (e.g. the sum
    // reduction reading the max result) are running accumulators broadcast over
    // the reduction axis; they need not appear in E, so skip them.
    if (innerResults.contains(in1Source))
      continue;
    OpOperand *inE = nullptr;
    for (OpOperand *cand : e.getDpsInputOperands()) {
      if (resolveSliceSource(cand->get()) == in1Source) {
        inE = cand;
        break;
      }
    }
    if (!inE) {
      LLVM_DEBUG(
          DBGS() << "checkInnerReductionAgainstElementwise: failed -- R1 "
                    "input is not also an input of E: "
                 << in1->get() << "\n");
      return failure();
    }
    AffineMap m1 = r1.getMatchingIndexingMap(in1);
    AffineMap mE = e.getMatchingIndexingMap(inE);
    if (m1.getNumResults() != mE.getNumResults()) {
      LLVM_DEBUG(
          DBGS() << "checkInnerReductionAgainstElementwise: failed -- shared "
                    "input has maps of different rank in R1 vs E (R1: "
                 << m1 << ", E: " << mE << ").\n");
      return failure();
    }
    for (auto [e1, e2] : llvm::zip_equal(m1.getResults(), mE.getResults())) {
      auto d1 = dyn_cast<AffineDimExpr>(e1);
      auto d2 = dyn_cast<AffineDimExpr>(e2);
      if (!d1 || !d2) {
        LLVM_DEBUG(DBGS() << "checkInnerReductionAgainstElementwise: failed -- "
                             "shared input map has a non-dim affine expr (R1: "
                          << m1 << ", E: " << mE << ").\n");
        return failure();
      }
      if (!tryAddMapping(d1.getPosition(), d2.getPosition())) {
        LLVM_DEBUG(DBGS() << "checkInnerReductionAgainstElementwise: failed -- "
                             "inconsistent dim mapping between R1 and E "
                             "derived from shared inputs (R1.d"
                          << d1.getPosition() << " -> {E.d"
                          << phi[d1.getPosition()] << ", E.d"
                          << d2.getPosition() << "}).\n");
        return failure();
      }
    }
  }
  // phi must be total over R1's loop dims so we can translate R1's init map
  // (and any other R1 map) into E's iter space during fusion.
  if (phi.size() != r1.getNumLoops()) {
    LLVM_DEBUG(
        DBGS()
        << "checkInnerReductionAgainstElementwise: failed -- derived dim "
           "mapping does not cover all of R1's loop dims (covered "
        << phi.size() << " of " << r1.getNumLoops() << ").\n");
    return failure();
  }
  auto redIt = phi.find(r1RedDims.front());
  if (redIt == phi.end() || redIt->second != eTiledDim) {
    LLVM_DEBUG(
        DBGS() << "checkInnerReductionAgainstElementwise: failed -- R1's "
                  "reduction dim is not aligned with the E dim carrying "
                  "R2's reduction axis under the derived dim mapping.\n");
    return failure();
  }
  return success();
}

/// Find the `e` loop dim carrying `r2`'s reduction axis, i.e. the axis along
/// which `e` must be re-sliced once fused. `E`'s output map and `R2`'s map for
/// `r2EOperand` describe the same tensor, so they align position-by-position;
/// the `e` dim at the position where `r2` reads `r2RedDim` is the answer.
/// Returns `nullopt` if either map is not a pure dim projection.
static std::optional<unsigned>
findElementwiseDimForR2ReductionDim(linalg::GenericOp e, linalg::GenericOp r2,
                                    OpOperand *r2EOperand, unsigned r2RedDim) {
  AffineMap r2Map = r2.getMatchingIndexingMap(r2EOperand);
  AffineMap eOutMap = e.getMatchingIndexingMap(e.getDpsInitOperand(0));
  if (r2Map.getNumResults() != eOutMap.getNumResults()) {
    LLVM_DEBUG(DBGS() << "findElementwiseDimForR2ReductionDim: failed -- R2's "
                         "map for E's result and E's output map have different "
                         "rank (R2: "
                      << r2Map << ", E out: " << eOutMap << ").\n");
    return std::nullopt;
  }
  for (auto [r2Expr, eExpr] :
       llvm::zip_equal(r2Map.getResults(), eOutMap.getResults())) {
    auto r2Dim = dyn_cast<AffineDimExpr>(r2Expr);
    auto eDim = dyn_cast<AffineDimExpr>(eExpr);
    if (!r2Dim || !eDim) {
      LLVM_DEBUG(DBGS() << "findElementwiseDimForR2ReductionDim: failed -- a "
                           "non-dim affine expr in R2's map for E's result or "
                           "in E's output map (R2: "
                        << r2Map << ", E out: " << eOutMap << ").\n");
      return std::nullopt;
    }
    if (r2Dim.getPosition() == r2RedDim)
      return eDim.getPosition();
  }
  LLVM_DEBUG(DBGS() << "findElementwiseDimForR2ReductionDim: failed -- R2's "
                       "reduction dim d"
                    << r2RedDim
                    << " does not appear in its map for E's result: " << r2Map
                    << "\n");
  return std::nullopt;
}

/// Whether to fuse a *clone* of `e` instead of `e` itself, i.e. whether `e` has
/// any consumer besides `r2`.
///
/// The fused `E` computes each tile against the *running* accumulator, which
/// the online correction only accounts for on `R2`'s accumulator. `E`'s own
/// full-extent loop result is therefore stale in all but the last tile -- for
/// softmax the tile at step `t` is `exp(x - m_t)`, off by `exp(m - m_t)` -- so
/// no outside consumer may read it. Cloning leaves the original in place, still
/// reading `R1`'s final result and recomputing the term correctly. This covers
/// softmax's normalizing divide as well as a second consumer reduction over the
/// same axis (an attention chain's row sum and `P @ V`), which is fused by a
/// further application. The clone's full-extent result is dead;
/// `--remove-dead-values` unwinds it.
static bool needsElementwiseClone(linalg::GenericOp e, linalg::GenericOp r2) {
  return !llvm::all_of(e.getResult(0).getUsers(), [&](Operation *user) {
    return user == r2.getOperation();
  });
}

/// Verify that the `R1 -> E -> R2` chain can be fused into `r1Loop`, whose
/// `step` is the tile size and `ub - lb` the full reduction extent.
/// `resultToInner` maps each loop result to the inner reduction producing it,
/// as built by `mapLoopResultsToInnerReductions`.
///
/// Requires: `E` all-parallel with one result; `R2` a single-result `addf`
/// reduction over one innermost axis with a zero init, exactly one input being
/// `E`'s result and every input carrying that axis. The per-inner-reduction
/// checks run against `E` for every loop result `E` consumes (see
/// `checkInnerReductionAgainstElementwise`). On success `eTiledDim` receives
/// the `E` loop dim carrying R2's reduction axis, which retiling needs.
static LogicalResult checkLegalFusionTriple(
    scf::ForOp r1Loop, ArrayRef<linalg::GenericOp> resultToInner,
    linalg::GenericOp e, linalg::GenericOp r2, unsigned &eTiledDim) {

  LLVM_DEBUG(
      DBGS() << "checkLegalFusionTriple: checking candidate:\n  R1 loop: "
             << *r1Loop << "\n  E: " << *e << "\n  R2: " << *r2 << "\n");
  if (r2->getNumResults() != 1 || r2.getNumDpsInits() != 1) {
    LLVM_DEBUG(DBGS() << "checkLegalFusionTriple: failed -- R2 does not have "
                         "exactly one result/init (results: "
                      << r2->getNumResults()
                      << ", inits: " << r2.getNumDpsInits() << ").\n");
    return failure();
  }

  // E must be all-parallel with one result: an elementwise term, not a
  // reduction.
  if (e->getNumResults() != 1 || e.getNumDpsInits() != 1) {
    LLVM_DEBUG(DBGS() << "checkLegalFusionTriple: failed -- E does not have "
                         "exactly one result/init (results: "
                      << e->getNumResults() << ", inits: " << e.getNumDpsInits()
                      << ").\n");
    return failure();
  }
  if (e.getNumReductionLoops() != 0) {
    LLVM_DEBUG(
        DBGS() << "checkLegalFusionTriple: failed -- E is not all-parallel "
                  "(it has "
               << e.getNumReductionLoops() << " reduction loops).\n");
    return failure();
  }

  // R2 may have several inputs (e.g. a contraction), but exactly one must be
  // E's result.
  if (e->getBlock() != r2->getBlock() || e->getBlock() != r1Loop->getBlock()) {
    LLVM_DEBUG(
        DBGS() << "checkLegalFusionTriple: failed -- R1 loop, E and R2 are "
                  "not all in the same block.\n");
    return failure();
  }
  OpOperand *r2EOperand = nullptr;
  for (OpOperand *in : r2.getDpsInputOperands()) {
    if (in->get() != e.getResult(0))
      continue;
    if (r2EOperand) {
      LLVM_DEBUG(DBGS() << "checkLegalFusionTriple: failed -- R2 consumes E's "
                           "result more than once.\n");
      return failure();
    }
    r2EOperand = in;
  }
  if (!r2EOperand) {
    LLVM_DEBUG(DBGS() << "checkLegalFusionTriple: failed -- no R2 input is E's "
                         "result.\n");
    return failure();
  }

  // R2 must have exactly one reduction loop, and it must be its innermost loop.
  SmallVector<unsigned> r2RedDims;
  r2.getReductionDims(r2RedDims);
  if (r2RedDims.size() != 1) {
    LLVM_DEBUG(DBGS() << "checkLegalFusionTriple: failed -- R2 does not have "
                         "exactly one reduction iterator ("
                      << r2RedDims.size() << ").\n");
    return failure();
  }
  if (r2RedDims.front() != r2.getNumLoops() - 1) {
    LLVM_DEBUG(
        DBGS() << "checkLegalFusionTriple: failed -- reduction iterator is "
                  "not the innermost loop in R2.\n");
    return failure();
  }

  // Every R2 input must carry R2's reduction axis. An input that does not is
  // broadcast across the reduction and would have to be re-derived per tile
  // rather than re-sliced, which the rewrite does not model.
  for (OpOperand *in : r2.getDpsInputOperands()) {
    AffineMap m = r2.getMatchingIndexingMap(in);
    bool carriesRedDim = false;
    for (AffineExpr expr : m.getResults()) {
      auto dimExpr = dyn_cast<AffineDimExpr>(expr);
      if (!dimExpr) {
        LLVM_DEBUG(
            DBGS() << "checkLegalFusionTriple: failed -- R2 input indexing "
                      "map has a non-dim affine expr: "
                   << m << "\n");
        return failure();
      }
      if (dimExpr.getPosition() == r2RedDims.front())
        carriesRedDim = true;
    }
    if (!carriesRedDim) {
      LLVM_DEBUG(DBGS() << "checkLegalFusionTriple: failed -- R2 input is not "
                           "reduced along R2's reduction axis (map does not "
                           "reference dim "
                        << r2RedDims.front() << "): " << m << "\n");
      return failure();
    }
  }

  // The E axis to re-slice once E is cloned into R1's loop, which R1's
  // reduction dim must align with.
  std::optional<unsigned> eRedDim =
      findElementwiseDimForR2ReductionDim(e, r2, r2EOperand, r2RedDims.front());
  if (!eRedDim)
    return failure();
  eTiledDim = *eRedDim;

  // Bounds must be constant, R2's reduction extent must equal the loop extent,
  // and the tile size must divide it evenly.
  std::optional<int64_t> lb = getConstantIntValue(r1Loop.getLowerBound());
  std::optional<int64_t> ub = getConstantIntValue(r1Loop.getUpperBound());
  std::optional<int64_t> step = getConstantIntValue(r1Loop.getStep());
  if (!lb || !ub || !step || *step <= 0) {
    LLVM_DEBUG(
        DBGS() << "checkLegalFusionTriple: failed -- R1 reduction loop does "
                  "not have constant, positive bounds/step.\n");
    return failure();
  }
  int64_t fullExtent = *ub - *lb;
  int64_t tileSize = *step;
  SmallVector<int64_t> ranges2 = r2.getStaticLoopRanges();
  int64_t r2RedRange = ranges2[r2RedDims.front()];
  if (ShapedType::isDynamic(r2RedRange)) {
    LLVM_DEBUG(
        DBGS() << "checkLegalFusionTriple: failed -- R2 reduction range is "
                  "dynamic; fusion requires a static reduction "
                  "extent.\n");
    return failure();
  }
  if (r2RedRange != fullExtent) {
    LLVM_DEBUG(
        DBGS() << "checkLegalFusionTriple: failed -- R2's reduction extent ("
               << r2RedRange << ") differs from the R1 loop extent ("
               << fullExtent << ").\n");
    return failure();
  }
  if (fullExtent % tileSize != 0) {
    LLVM_DEBUG(DBGS() << "checkLegalFusionTriple: failed -- tile size "
                      << tileSize
                      << " does not evenly divide the reduction extent "
                      << fullExtent << ".\n");
    return failure();
  }

  // E's extent along that axis must match too, so re-slicing is well defined.
  SmallVector<int64_t> rangesE = e.getStaticLoopRanges();
  int64_t eRedRange = rangesE[*eRedDim];
  if (ShapedType::isDynamic(eRedRange) || eRedRange != fullExtent) {
    LLVM_DEBUG(
        DBGS() << "checkLegalFusionTriple: failed -- E's extent along the "
                  "axis carrying R2's reduction ("
               << eRedRange << ") differs from the R1 loop extent ("
               << fullExtent << ").\n");
    return failure();
  }

  // At least one E operand must read a loop result: that is the R1 -> E
  // dependence making the chain fusable.
  R1AsElementwiseInputInfo r1AsE = collectR1AsElementwiseInputs(r1Loop, e);
  if (r1AsE.operands.empty()) {
    LLVM_DEBUG(DBGS() << "checkLegalFusionTriple: failed -- E does not consume "
                         "any result of the R1 loop.\n");
    return failure();
  }

  // Inner reduction results, used to skip sibling accumulators in the phi
  // check.
  llvm::SmallDenseSet<Value, 4> innerResults;
  for (linalg::GenericOp inner : resultToInner)
    if (inner)
      innerResults.insert(inner.getResult(0));

  // Each R1 result E reads must be broadcast across `eRedDim`, i.e. its map
  // must not reference it. That is what makes the accumulator a
  // per-parallel-slice value the correction can rescale.
  for (OpOperand *r1AsEInput : r1AsE.operands) {
    AffineMap r1AsEMap = e.getMatchingIndexingMap(r1AsEInput);
    for (AffineExpr expr : r1AsEMap.getResults()) {
      auto dimExpr = dyn_cast<AffineDimExpr>(expr);
      if (!dimExpr) {
        LLVM_DEBUG(DBGS() << "checkLegalFusionTriple: failed -- R1-as-E-input "
                             "indexing map has a non-dim affine expr: "
                          << r1AsEMap << "\n");
        return failure();
      }
      if (dimExpr.getPosition() == *eRedDim) {
        LLVM_DEBUG(
            DBGS() << "checkLegalFusionTriple: failed -- R1's result is not "
                      "broadcast across the E axis carrying R2's reduction "
                      "(map references dim "
                   << dimExpr.getPosition() << "): " << r1AsEMap << "\n");
        return failure();
      }
    }
  }

  // Check every loop result E consumes against its inner reduction.
  for (auto [operand, resultIdx] :
       llvm::zip_equal(r1AsE.operands, r1AsE.r1ResultIdx)) {
    linalg::GenericOp inner = resultToInner[resultIdx];
    if (!inner) {
      LLVM_DEBUG(
          DBGS() << "checkLegalFusionTriple: failed -- consumed loop result "
                 << resultIdx
                 << " is not produced by an inner reduction generic.\n");
      return failure();
    }
    if (failed(checkInnerReductionAgainstElementwise(inner, e, *eRedDim,
                                                     innerResults)))
      return failure();
  }

  // R2's region must be a single-combiner `addf` reduction with a zero init,
  // over a supported float type: that is what makes rescaling a valid repair.
  SmallVector<Operation *> combiners;
  if (!matchReduction(r2.getRegionOutputArgs(), /*redPos=*/0, combiners)) {
    LLVM_DEBUG(
        DBGS() << "checkLegalFusionTriple: failed -- R2's region does not "
                  "match a reduction pattern.\n");
    return failure();
  }
  if (combiners.size() != 1) {
    LLVM_DEBUG(DBGS() << "checkLegalFusionTriple: failed -- R2's reduction has "
                      << combiners.size()
                      << " combiners, expected exactly 1.\n");
    return failure();
  }
  if (!isa<arith::AddFOp>(combiners.front())) {
    LLVM_DEBUG(
        DBGS() << "checkLegalFusionTriple: failed -- R2's combiner is not "
                  "arith.addf: "
               << *combiners.front() << "\n");
    return failure();
  }

  Value r2Init = r2.getDpsInitOperand(0)->get();
  if (!isDefinedAsZero(r2Init)) {
    LLVM_DEBUG(
        DBGS() << "checkLegalFusionTriple: failed -- R2's init is not the "
                  "additive identity (zero): "
               << r2Init << "\n");
    return failure();
  }

  Type elt = getElementTypeOrSelf(r2->getResultTypes().front());
  if (!(elt.isF16() || elt.isBF16() || elt.isF32() || elt.isF64())) {
    LLVM_DEBUG(DBGS() << "checkLegalFusionTriple: failed -- R2's element type "
                      << elt
                      << " is not a supported floating-point type "
                         "(f16/bf16/f32/f64).\n");
    return failure();
  }

  // Any other user of an R1 loop result must post-dominate E, since fusing
  // moves the loop to just before E. There is no matching condition on E's
  // users: any user besides R2 makes the fusion clone E and leave the original
  // -- and those users -- in place (see `needsElementwiseClone`).
  PostDominanceInfo postDom;
  for (Value r1Result : r1Loop->getResults()) {
    for (Operation *user : r1Result.getUsers()) {
      if (user == e.getOperation())
        continue;
      if (!postDom.postDominates(user, e.getOperation())) {
        LLVM_DEBUG(DBGS() << "checkLegalFusionTriple: failed -- user of an R1 "
                             "result does not post-dominate E: "
                          << *user << "\n");
        return failure();
      }
    }
  }

  return success();
}

/// Cut every `offsets`/`sizes` position whose map result references `tiledDim`
/// down to the current tile (offset = IV, size = `tileSize`). Positions already
/// at `tileSize` are left alone. Returns true if anything changed.
static bool retileAlongDim(AffineMap map, unsigned tiledDim, Value iv,
                           int64_t tileSize,
                           SmallVectorImpl<OpFoldResult> &offsets,
                           SmallVectorImpl<OpFoldResult> &sizes, OpBuilder &b) {
  bool changed = false;
  for (auto [i, expr] : llvm::enumerate(map.getResults())) {
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (!dimExpr || dimExpr.getPosition() != tiledDim)
      continue;
    if (isConstantIntValue(sizes[i], tileSize))
      continue;
    offsets[i] = iv;
    sizes[i] = b.getIndexAttr(tileSize);
    changed = true;
  }
  return changed;
}

/// Re-slice `fused` along `tiledDim` down to the tile of `redLoop`.
///
/// `tileAndFuseConsumer` only restricts the dims a consumer shares with the
/// loop's yielded slices, and R1's result carries no reduction axis, so the
/// fused ops still span the whole reduction extent (e.g. `[64, 512]` rather
/// than one
/// `[64, 32]` tile). Every operand fed by a full-extent `tensor.extract_slice`
/// whose map references `tiledDim` is cut here. For an all-parallel `E` that
/// includes its destination, so the result type shrinks: the op is rebuilt and
/// the `tensor.insert_slice` yielding it re-sliced to match. For a reduction
/// `R2` the output map never references `tiledDim`, so only inputs change.
static FailureOr<linalg::GenericOp>
retileFusedConsumerToTile(RewriterBase &rewriter, linalg::GenericOp fused,
                          unsigned tiledDim, scf::ForOp redLoop,
                          int64_t tileSize) {
  Value iv = redLoop.getInductionVar();

  // Re-slice every operand (inputs and inits) fed by a full-extent slice.
  bool initChanged = false;
  for (OpOperand &operand : fused->getOpOperands()) {
    auto sliceOp = operand.get().getDefiningOp<tensor::ExtractSliceOp>();
    if (!sliceOp)
      continue;
    AffineMap map = fused.getMatchingIndexingMap(&operand);
    SmallVector<OpFoldResult> offsets = sliceOp.getMixedOffsets();
    SmallVector<OpFoldResult> sizes = sliceOp.getMixedSizes();
    SmallVector<OpFoldResult> strides = sliceOp.getMixedStrides();
    if (!retileAlongDim(map, tiledDim, iv, tileSize, offsets, sizes, rewriter))
      continue;

    rewriter.setInsertionPoint(sliceOp);
    auto retiled = tensor::ExtractSliceOp::create(rewriter, sliceOp.getLoc(),
                                                  sliceOp.getSource(), offsets,
                                                  sizes, strides);
    rewriter.replaceOp(sliceOp, retiled.getResult());
    if (fused.isDpsInit(&operand))
      initChanged = true;
    LLVM_DEBUG(DBGS() << "retileFusedConsumerToTile: retiled operand slice to: "
                      << retiled << "\n");
  }

  // A reduction consumer's output does not span `tiledDim`, so its result type
  // is unchanged and there is nothing further to do.
  if (!initChanged)
    return fused;

  // The destination shrank, so rebuild the op to pick up the smaller result
  // type, moving its body across unchanged.
  rewriter.setInsertionPoint(fused);
  auto retiledOp = linalg::GenericOp::create(
      rewriter, fused.getLoc(),
      TypeRange{fused.getDpsInits().front().getType()}, fused.getDpsInputs(),
      fused.getDpsInits(), fused.getIndexingMapsArray(),
      fused.getIteratorTypesArray());
  rewriter.inlineRegionBefore(fused.getRegion(), retiledOp.getRegion(),
                              retiledOp.getRegion().begin());

  // Re-slice the `tensor.insert_slice` that yields the (now smaller) result
  // into the loop-carried destination.
  for (OpOperand &use :
       llvm::make_early_inc_range(fused.getResult(0).getUses())) {
    auto insertOp = dyn_cast<tensor::InsertSliceOp>(use.getOwner());
    if (!insertOp)
      continue;
    SmallVector<OpFoldResult> offsets = insertOp.getMixedOffsets();
    SmallVector<OpFoldResult> sizes = insertOp.getMixedSizes();
    SmallVector<OpFoldResult> strides = insertOp.getMixedStrides();
    AffineMap outMap = fused.getMatchingIndexingMap(fused.getDpsInitOperand(0));
    retileAlongDim(outMap, tiledDim, iv, tileSize, offsets, sizes, rewriter);
    rewriter.setInsertionPoint(insertOp);
    auto retiledInsert = tensor::InsertSliceOp::create(
        rewriter, insertOp.getLoc(), retiledOp.getResult(0), insertOp.getDest(),
        offsets, sizes, strides);
    rewriter.replaceOp(insertOp, retiledInsert.getResult());
    LLVM_DEBUG(DBGS() << "retileFusedConsumerToTile: retiled insert_slice to: "
                      << retiledInsert << "\n");
  }

  rewriter.replaceOp(fused, retiledOp.getResults());
  return retiledOp;
}

/// A running accumulator as seen by the fused `E`: the input operand reading
/// its new (current-tile) value, paired with the old (previous running) value.
struct ConsumedAccumulator {
  OpOperand *use; // fused E input operand; use->get() is the new value.
  Value oldValue; // previous running value (the inner reduction's DPS init).
};

/// The constant that eliminates one operand's contribution to `op`, making it
/// the identity on the other: `0` for `addf`/`subf`/`addi`/`subi`, `1` for
/// `mulf`/`divf`/`muli`. Unlike `arith::getNeutralElement` this is defined for
/// the non-commutative ops correction terms contain. `nullopt` if there is
/// none.
static std::optional<TypedAttr> getOperandEliminatingConstant(Operation *op) {
  Type elt = getElementTypeOrSelf(op->getResult(0).getType());
  auto fp = [&](double v) -> std::optional<TypedAttr> {
    return cast<TypedAttr>(FloatAttr::get(elt, v));
  };
  auto in = [&](int64_t v) -> std::optional<TypedAttr> {
    return cast<TypedAttr>(IntegerAttr::get(elt, v));
  };
  return llvm::TypeSwitch<Operation *, std::optional<TypedAttr>>(op)
      .Case<arith::AddFOp, arith::SubFOp>([&](auto) { return fp(0.0); })
      .Case<arith::MulFOp, arith::DivFOp>([&](auto) { return fp(1.0); })
      .Case<arith::AddIOp, arith::SubIOp>([&](auto) { return in(0); })
      .Case<arith::MulIOp>([&](auto) { return in(1); })
      .Default([](Operation *) { return std::nullopt; });
}

/// Emit `E`'s body isolated on its accumulator inputs, with every data input
/// neutralized at its use point. `uses` binds each accumulator-reading operand
/// to the value to substitute (the new value for one factor, the old for the
/// other).
///
/// `E` being a separate op, its body can be cloned directly; its yielded value
/// is the term. Data block arguments are replaced per use by
/// `getOperandEliminatingConstant`, so each data op drops out instead of
/// contributing a magnitude. Sound because the data inputs cancel in the
/// new/old ratio: for `E = exp(x - m)`, `exp(x - m_new)/exp(x - m_old)` is
/// independent of `x`. Returns the cloned term, or null on failure.
static Value emitCorrectionTerm(OpBuilder &b, Location loc, linalg::GenericOp e,
                                ArrayRef<std::pair<OpOperand *, Value>> uses) {
  auto yieldOp = cast<linalg::YieldOp>(e.getBlock()->getTerminator());
  assert(yieldOp.getNumOperands() == 1 && "expected E to yield a single value");
  Value term = yieldOp.getOperand(0);

  // Bind the accumulator block arguments; data arguments stay unmapped and are
  // neutralized per consuming op below.
  IRMapping mapping;
  for (auto [use, value] : uses)
    mapping.map(e.getMatchingBlockArgument(use), value);

  for (Operation &op : e.getBlock()->without_terminator()) {
    SmallVector<BlockArgument> tempMapped;
    for (Value operand : op.getOperands()) {
      auto barg = dyn_cast<BlockArgument>(operand);
      if (!barg || barg.getOwner() != e.getBlock() || mapping.contains(barg))
        continue;
      std::optional<TypedAttr> neutral = getOperandEliminatingConstant(&op);
      if (!neutral)
        return nullptr;
      mapping.map(barg, arith::ConstantOp::create(b, loc, *neutral));
      tempMapped.push_back(barg);
    }
    b.clone(op, mapping);
    // Drop the per-op substitutions so the next consumer of the same data
    // argument gets its own neutral element.
    for (BlockArgument barg : tempMapped)
      mapping.erase(barg);
  }
  return mapping.lookupOrDefault(term);
}

/// Build the per-tile rescale factor `term(new) / term(old)`, where `term` is
/// `E` isolated on the accumulators it reads (see `emitCorrectionTerm`).
///
/// Both evaluations and the division are `linalg.generic`s over the accumulator
/// iteration space, giving the factor `r2`'s output shape. Each accumulator is
/// read through its `E` map with `eTiledDim` projected out; legality guarantees
/// those maps do not reference it, so the projection is lossless. Null on
/// failure.
static Value calculateCorrectionFactor(PatternRewriter &rewriter,
                                       linalg::GenericOp e,
                                       linalg::GenericOp r2, unsigned eTiledDim,
                                       ArrayRef<ConsumedAccumulator> accs) {
  Location loc = r2.getLoc();

  // Built over `r2`'s parallel iteration space, which for a contraction is
  // wider than `E`'s (e.g. GEMM's N dim) and which the accumulator broadcasts
  // over. Translate each accumulator map from `E`'s dims into `r2`'s by
  // aligning `E`'s output map with `r2`'s map for `E`'s result
  // position-by-position.
  OpOperand *r2EOperand = nullptr;
  for (OpOperand *in : r2.getDpsInputOperands())
    if (in->get() == e.getResult(0))
      r2EOperand = in;
  if (!r2EOperand)
    return nullptr;
  AffineMap eOutMap = e.getMatchingIndexingMap(e.getDpsInitOperand(0));
  AffineMap r2EMap = r2.getMatchingIndexingMap(r2EOperand);
  if (eOutMap.getNumResults() != r2EMap.getNumResults())
    return nullptr;
  SmallVector<AffineExpr> eDimToR2(e.getNumLoops());
  for (auto [eExpr, r2Expr] :
       llvm::zip_equal(eOutMap.getResults(), r2EMap.getResults())) {
    auto eDim = dyn_cast<AffineDimExpr>(eExpr);
    auto r2Dim = dyn_cast<AffineDimExpr>(r2Expr);
    if (!eDim || !r2Dim)
      return nullptr;
    eDimToR2[eDim.getPosition()] = r2Dim;
  }
  // Unmapped E dims cannot appear in an accumulator map; a placeholder turns
  // any violation into a failed projection rather than a miscompile.
  for (AffineExpr &expr : eDimToR2)
    if (!expr)
      expr = rewriter.getAffineConstantExpr(0);

  SmallVector<unsigned> r2RedDims;
  r2.getReductionDims(r2RedDims);
  llvm::SmallBitVector r2Projected(r2.getNumLoops());
  r2Projected.set(r2RedDims.front());

  SmallVector<AffineMap> indexingMaps;
  for (const ConsumedAccumulator &acc : accs) {
    AffineMap inR2 = e.getMatchingIndexingMap(acc.use).replaceDimsAndSymbols(
        eDimToR2, /*symReplacements=*/{}, r2.getNumLoops(),
        /*numResultSyms=*/0);
    indexingMaps.push_back(
        projectDims(inR2, r2Projected, /*compressDimsFlag=*/true));
  }

  // The factor takes R2's accumulator shape, identity-mapped.
  Value r2Acc = r2.getDpsInitOperand(0)->get();
  auto accType = cast<RankedTensorType>(r2Acc.getType());
  indexingMaps.push_back(rewriter.getMultiDimIdentityMap(accType.getRank()));
  if (llvm::any_of(indexingMaps, [&](AffineMap m) {
        return m.getNumDims() != accType.getRank();
      }))
    return nullptr;
  SmallVector<utils::IteratorType> iterTypes(accType.getRank(),
                                             utils::IteratorType::parallel);

  SmallVector<OpFoldResult> sizes = tensor::getMixedSizes(rewriter, loc, r2Acc);
  Value init =
      tensor::EmptyOp::create(rewriter, loc, sizes, accType.getElementType());

  // Replay E's body with each accumulator bound to the value chosen by `pick`.
  bool bodyFailed = false;
  auto buildTerm = [&](function_ref<Value(const ConsumedAccumulator &)> pick) {
    SmallVector<Value> ins = llvm::map_to_vector(
        accs, [&](const ConsumedAccumulator &acc) { return pick(acc); });
    return linalg::GenericOp::create(
        rewriter, loc, init.getType(), ins, ValueRange{init}, indexingMaps,
        iterTypes, [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
          SmallVector<std::pair<OpOperand *, Value>> uses;
          for (auto [acc, arg] : llvm::zip_equal(accs, args.drop_back()))
            uses.emplace_back(acc.use, arg);
          Value term = emitCorrectionTerm(b, nestedLoc, e, uses);
          if (!term) {
            bodyFailed = true;
            term = args.back();
          }
          linalg::YieldOp::create(b, nestedLoc, term);
        });
  };
  auto termNew =
      buildTerm([](const ConsumedAccumulator &acc) { return acc.use->get(); });
  auto termOld =
      buildTerm([](const ConsumedAccumulator &acc) { return acc.oldValue; });
  if (bodyFailed)
    return nullptr;

  SmallVector<Attribute> idMaps = llvm::map_to_vector(
      ElementwiseOp::getDefaultIndexingMaps(/*numMaps=*/3, accType.getRank(),
                                            rewriter.getContext()),
      [](AffineMap m) -> Attribute { return AffineMapAttr::get(m); });
  auto div = ElementwiseOp::create(
      rewriter, loc, ValueRange{termNew.getResult(0), termOld.getResult(0)},
      ValueRange{init},
      ElementwiseKindAttr::get(rewriter.getContext(), ElementwiseKind::div),
      /*indexingMaps=*/rewriter.getArrayAttr(idMaps));
  return div.getResult(0);
}

/// Rescale `fusedR2`'s running accumulator by the per-tile factor, keeping
/// partial sums from earlier tiles valid when `R1`'s accumulator changes: its
/// DPS init becomes `init * term(new)/term(old)`. The new values are `fusedE`'s
/// inputs; the old ones come from `newToOld` below.
static LogicalResult correctFusedR2Accumulator(PatternRewriter &rewriter,
                                               linalg::GenericOp fusedE,
                                               linalg::GenericOp fusedR2,
                                               unsigned eTiledDim) {
  auto loop = fusedR2->getParentOfType<scf::ForOp>();
  assert(loop && "expected fusedR2 to live inside the reduction loop");

  // Current-tile value -> previous running value, the latter being the inner
  // reduction's own DPS init. Using the init rather than the raw `iter_arg`
  // keeps the old value readable after the reduction has run: it is an
  // immutable SSA value, so bufferization copies the accumulator instead of
  // writing in place.
  llvm::SmallDenseMap<Value, Value, 4> newToOld;
  auto yieldOp = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  for (Value yielded : yieldOp.getOperands()) {
    auto insertSlice = yielded.getDefiningOp<tensor::InsertSliceOp>();
    if (!insertSlice)
      continue;
    auto inner = insertSlice.getSource().getDefiningOp<linalg::GenericOp>();
    if (!inner || inner.getNumDpsInits() != 1)
      continue;
    newToOld[insertSlice.getSource()] = inner.getDpsInitOperand(0)->get();
  }

  SmallVector<ConsumedAccumulator> accs;
  for (OpOperand *in : fusedE.getDpsInputOperands()) {
    auto it = newToOld.find(in->get());
    if (it != newToOld.end())
      accs.push_back({in, it->second});
  }
  if (accs.empty()) {
    LLVM_DEBUG(
        DBGS() << "correctFusedR2Accumulator: failed -- fused E does not "
                  "consume a running accumulator of the loop.\n");
    return failure();
  }

  rewriter.setInsertionPoint(fusedR2);
  Value factor =
      calculateCorrectionFactor(rewriter, fusedE, fusedR2, eTiledDim, accs);
  if (!factor) {
    LLVM_DEBUG(DBGS() << "correctFusedR2Accumulator: failed -- could not build "
                         "the correction factor from E.\n");
    return failure();
  }

  // Accumulator, factor and result share a shape, so identity maps suffice.
  Value acc = fusedR2.getDpsInitOperand(0)->get();
  auto accType = cast<RankedTensorType>(acc.getType());
  SmallVector<Attribute> idMaps = llvm::map_to_vector(
      ElementwiseOp::getDefaultIndexingMaps(/*numMaps=*/3, accType.getRank(),
                                            rewriter.getContext()),
      [](AffineMap m) -> Attribute { return AffineMapAttr::get(m); });
  auto mul = ElementwiseOp::create(
      rewriter, fusedR2.getLoc(), ValueRange{acc, factor}, ValueRange{acc},
      ElementwiseKindAttr::get(rewriter.getContext(), ElementwiseKind::mul),
      /*indexingMaps=*/rewriter.getArrayAttr(idMaps));
  rewriter.modifyOpInPlace(
      fusedR2, [&]() { fusedR2.getDpsInitOperand(0)->set(mul.getResult(0)); });
  return success();
}

/// Whether `value` transitively depends on `op`. Distinguishes operands that
/// can be hoisted above the reduction loop from those computed by it.
static bool dependsOnOp(Value value, Operation *op) {
  if (value.getDefiningOp() == op)
    return true;
  SetVector<Operation *> slice;
  if (failed(getBackwardSlice(value, &slice)))
    // Be conservative: an unknown slice may well contain `op`.
    return true;
  return slice.contains(op);
}

/// Hoist `consumer`'s operand definitions above `insertionPoint` (the reduction
/// loop) where they are not already.
///
/// `scf::tileAndFuseConsumer` moves the loop to just before the consumer being
/// fused, and its dominance check rejects that when a consumer operand is
/// defined below the loop. Tiling routinely leaves such operands: `E`'s and
/// `R2`'s destinations, and the `tensor.extract_slice` feeding a data input
/// when an enclosing `scf.forall` tiled a parallel dim. Operands that depend on
/// the loop are skipped -- moving them would move the loop above itself.
static LogicalResult hoistConsumerOperandsBefore(RewriterBase &rewriter,
                                                 linalg::GenericOp consumer,
                                                 Operation *insertionPoint) {
  DominanceInfo dominance(insertionPoint->getParentOp());
  SmallVector<Value> toHoist;
  for (Value operand : consumer->getOperands()) {
    Operation *defOp = operand.getDefiningOp();
    if (!defOp || dominance.properlyDominates(defOp, insertionPoint))
      continue;
    if (dependsOnOp(operand, insertionPoint))
      continue;
    toHoist.push_back(operand);
  }
  if (toHoist.empty())
    return success();

  if (failed(moveValueDefinitions(rewriter, toHoist, insertionPoint))) {
    LLVM_DEBUG(
        DBGS() << "hoistConsumerOperandsBefore: failed -- could not move "
                  "operand definitions of "
               << *consumer.getOperation() << " before the loop.\n");
    return failure();
  }
  LLVM_DEBUG(DBGS() << "hoistConsumerOperandsBefore: hoisted " << toHoist.size()
                    << " operand definition(s) above the loop.\n");
  return success();
}

/// Fuse `e`, then `r2`, into the already-tiled producer loop `r1Loop`.
///
/// Both go through `scf::tileAndFuseConsumer`, which clones the consumer inside
/// the loop to compute one tile of its result. `E` is fused first so its result
/// becomes a loop result `R2` can be fused against. Operand definitions are
/// hoisted above the loop first (`hoistConsumerOperandsBefore`), each fused
/// clone is re-sliced to the tile (`retileFusedConsumerToTile`, along
/// `eTiledDim` for `E` and its own reduction dim for `R2`), and finally the
/// correction rescales `R2`'s accumulator (`correctFusedR2Accumulator`).
static FailureOr<scf::SCFFuseConsumerOfSliceResult>
fuseElementwiseAndR2IntoTiledR1Loop(PatternRewriter &rewriter,
                                    linalg::GenericOp e, linalg::GenericOp r2,
                                    scf::ForOp r1Loop, unsigned eTiledDim,
                                    int64_t tileSize) {
  // Fuse a clone when `E` has other consumers (see `needsElementwiseClone`).
  // The clone goes immediately before `E`, keeping it above every other user of
  // an R1 loop result so the loop can still move to just before the op being
  // fused.
  if (needsElementwiseClone(e, r2)) {
    OpOperand *r2EOperand = nullptr;
    for (OpOperand *in : r2.getDpsInputOperands())
      if (in->get() == e.getResult(0))
        r2EOperand = in;
    assert(r2EOperand && "legality guarantees exactly one R2 input reads E");
    rewriter.setInsertionPoint(e);
    auto eClone = cast<linalg::GenericOp>(rewriter.clone(*e.getOperation()));
    rewriter.modifyOpInPlace(r2,
                             [&]() { r2EOperand->set(eClone.getResult(0)); });
    LLVM_DEBUG(DBGS() << "fuseElementwiseAndR2IntoTiledR1Loop: E feeds another "
                         "consumer reduction; fusing a clone of it instead: "
                      << *eClone.getOperation() << "\n");
    e = eClone;
  }

  // Hoist both consumers' operands up front, so an unhoistable `R2` is rejected
  // before `E` has already been fused.
  for (linalg::GenericOp consumer : {e, r2})
    if (failed(hoistConsumerOperandsBefore(rewriter, consumer,
                                           r1Loop.getOperation())))
      return failure();

  SmallVector<LoopLikeOpInterface> loops = {
      cast<LoopLikeOpInterface>(r1Loop.getOperation())};

  SmallVector<unsigned> r2RedDims;
  r2.getReductionDims(r2RedDims);
  assert(r2RedDims.size() == 1 && "legality guarantees a single R2 reduction");

  // `tileAndFuseConsumer` updates `loops` in place with the rewritten loop.
  FailureOr<scf::SCFFuseConsumerOfSliceResult> fuseResult;
  SmallVector<linalg::GenericOp> fusedOps; // [0] = fused E, [1] = fused R2.
  for (auto [consumer, tiledDim] :
       {std::pair{e, eTiledDim}, std::pair{r2, r2RedDims.front()}}) {
    rewriter.setInsertionPoint(consumer);
    fuseResult = scf::tileAndFuseConsumer(rewriter, consumer.getOperation(),
                                          loops, /*fn=*/nullptr);
    if (failed(fuseResult)) {
      LLVM_DEBUG(DBGS() << "fuseElementwiseAndR2IntoTiledR1Loop: failed -- "
                           "tileAndFuseConsumer did not succeed for "
                        << *consumer.getOperation() << "\n");
      return failure();
    }
    if (fuseResult->tiledOps.empty()) {
      LLVM_DEBUG(DBGS() << "fuseElementwiseAndR2IntoTiledR1Loop: failed -- "
                           "tileAndFuseConsumer fused nothing for "
                        << *consumer.getOperation() << "\n");
      return failure();
    }

    LLVM_DEBUG(
        DBGS() << "=== after fusing " << consumer->getName()
               << " into the tiled R1 loop ===\n"
               << *fuseResult->tiledOps.front()
                       ->getParentWithTrait<OpTrait::IsIsolatedFromAbove>()
               << "\n");

    // Cut the fused clone down to the current reduction tile.
    auto fused = dyn_cast<linalg::GenericOp>(fuseResult->tiledOps.front());
    if (!fused) {
      LLVM_DEBUG(
          DBGS() << "fuseElementwiseAndR2IntoTiledR1Loop: failed -- fused "
                    "op is not a linalg.generic.\n");
      return failure();
    }
    auto redLoop = cast<scf::ForOp>(loops.front().getOperation());
    FailureOr<linalg::GenericOp> retiled =
        retileFusedConsumerToTile(rewriter, fused, tiledDim, redLoop, tileSize);
    if (failed(retiled))
      return failure();
    // Retiling may rebuild the op (E's result type shrinks), and the correction
    // reads the fused E's operands, so track the current one.
    fusedOps.push_back(*retiled);

    LLVM_DEBUG(
        DBGS() << "=== after re-slicing the fused " << consumer->getName()
               << " to the reduction tile ===\n"
               << *redLoop->getParentWithTrait<OpTrait::IsIsolatedFromAbove>()
               << "\n");
  }

  if (failed(correctFusedR2Accumulator(rewriter, fusedOps[0], fusedOps[1],
                                       eTiledDim)))
    return failure();

  LLVM_DEBUG(DBGS() << "=== after applying the online correction ===\n"
                    << *loops.front()
                            ->getParentWithTrait<OpTrait::IsIsolatedFromAbove>()
                    << "\n");

  return fuseResult;
}

/// Matches the chain bottom-up from the consumer reduction `R2`: one of its
/// inputs must come from an all-parallel `E`, one of whose inputs must come
/// from a
/// `__reduction_loop__`-marked `scf.for` (the already-tiled `R1`). If the
/// triple is legal, the chain is fused into that loop; the tile size is the
/// loop's `step`, as this pattern does not tile `R1` itself.
struct FuseDependentReductionsPattern
    : public OpRewritePattern<linalg::GenericOp> {
  FuseDependentReductionsPattern(MLIRContext *context, ControlFusionFn fun,
                                 PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::GenericOp>(context, benefit),
        controlFn(std::move(fun)) {}

  LogicalResult matchAndRewrite(linalg::GenericOp r2,
                                PatternRewriter &rewriter) const override {
    LLVM_DEBUG(
        DBGS() << "FuseDependentReductionsPattern: considering candidate R2: "
               << *r2.getOperation() << "\n");
    if (r2.getNumReductionLoops() == 0) {
      LLVM_DEBUG(DBGS() << "FuseDependentReductionsPattern: failed -- R2 has "
                           "no reduction loops: "
                        << *r2.getOperation() << "\n");
      return failure();
    }

    // Exactly one R2 input must come from an all-parallel `linalg.generic`;
    // reject an ambiguous match against two distinct candidates.
    linalg::GenericOp e;
    for (OpOperand *in : r2.getDpsInputOperands()) {
      auto candidateE = in->get().getDefiningOp<linalg::GenericOp>();
      if (!candidateE || candidateE.getNumReductionLoops() != 0)
        continue;
      if (!controlFn(in)) {
        LLVM_DEBUG(
            DBGS() << "FuseDependentReductionsPattern: skipping candidate "
                      "E -- control function vetoed fusing R2 with it.\n");
        continue;
      }
      if (e && candidateE != e) {
        LLVM_DEBUG(
            DBGS() << "FuseDependentReductionsPattern: failed -- more than "
                      "one R2 input is produced by an all-parallel "
                      "elementwise linalg.generic; the chain is "
                      "ambiguous.\n");
        return failure();
      }
      e = candidateE;
    }
    if (!e) {
      LLVM_DEBUG(
          DBGS() << "FuseDependentReductionsPattern: failed -- no R2 input "
                    "is produced by an all-parallel elementwise "
                    "linalg.generic.\n");
      return failure();
    }
    LLVM_DEBUG(
        DBGS() << "FuseDependentReductionsPattern: found elementwise term "
                  "E: "
               << *e.getOperation() << "\n");

    // Try each E input as a candidate R1: a `__reduction_loop__`-marked
    // `scf.for` in the same block whose body holds a reduction generic, forming
    // a legal triple. First success wins.
    scf::ForOp r1Loop;
    unsigned eTiledDim = 0;
    for (OpOperand *in : e.getDpsInputOperands()) {
      auto candidateLoop = in->get().getDefiningOp<scf::ForOp>();
      if (!candidateLoop || !candidateLoop->hasAttr(kReductionLoopAttrName))
        continue;
      if (!controlFn(in)) {
        LLVM_DEBUG(DBGS() << "FuseDependentReductionsPattern: skipping "
                             "candidate R1 -- control function vetoed fusing "
                             "through this operand.\n");
        continue;
      }
      if (candidateLoop->getBlock() != e->getBlock()) {
        LLVM_DEBUG(DBGS() << "FuseDependentReductionsPattern: skipping "
                             "candidate R1 -- not in the same block as E.\n");
        continue;
      }
      if (collectInnerReductionGenerics(candidateLoop).empty()) {
        LLVM_DEBUG(DBGS() << "FuseDependentReductionsPattern: skipping "
                             "candidate R1 -- loop body does not contain any "
                             "reduction linalg.generic.\n");
        continue;
      }
      SmallVector<linalg::GenericOp> resultToInner =
          mapLoopResultsToInnerReductions(candidateLoop);
      if (failed(checkLegalFusionTriple(candidateLoop, resultToInner, e, r2,
                                        eTiledDim))) {
        LLVM_DEBUG(DBGS() << "FuseDependentReductionsPattern: skipping "
                             "candidate R1 -- checkLegalFusionTriple rejected "
                             "the candidate.\n");
        continue;
      }
      r1Loop = candidateLoop;
      break;
    }
    if (!r1Loop) {
      LLVM_DEBUG(DBGS() << "FuseDependentReductionsPattern: failed -- no DPS "
                           "input of E is an already-tiled fusable producer "
                           "reduction: "
                        << *r2.getOperation() << "\n");
      return failure();
    }
    LLVM_DEBUG(DBGS() << "FuseDependentReductionsPattern: selected R1 loop: "
                      << *r1Loop.getOperation() << "\n");

    // Verified constant and positive by `checkLegalFusionTriple`.
    int64_t tileSize = *getConstantIntValue(r1Loop.getStep());

    if (failed(fuseElementwiseAndR2IntoTiledR1Loop(rewriter, e, r2, r1Loop,
                                                   eTiledDim, tileSize)))
      return failure();

    return success();
  }

private:
  ControlFusionFn controlFn;
};

} // namespace

void mlir::linalg::populateDependantReductionFusionPatterns(
    RewritePatternSet &patterns,
    const ControlFusionFn &controlDependantReductionFusion) {
  patterns.add<FuseDependentReductionsPattern>(patterns.getContext(),
                                               controlDependantReductionFusion);
}
