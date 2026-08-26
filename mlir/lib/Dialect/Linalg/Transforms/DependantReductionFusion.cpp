//===- DependantReductionFusion.cpp - Fuse dependent reductions -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pattern that fuses dependent reduction operations
// that share a reduction dimension into a single reduction. The canonical use
// case is converting two-pass softmax (max reduction followed by sum-of-exp
// reduction) into the online (one-pass) form.
//
// The chain is `R1 -> E -> R2`, with the elementwise term left UNFUSED:
//   R1  a reduction already tiled into a `__reduction_loop__` `scf.for`,
//   E   an all-parallel elementwise op consuming R1's result plus the data
//       inputs it shares with R1,
//   R2  a reduction over E's result, possibly alongside other operands (e.g. a
//       GEMM-like contraction), all reduced along R1's reduction axis.
// `E` and `R2` are cloned into R1's loop, re-sliced to the reduction tile, and
// R2's running accumulator is rescaled per tile by a correction factor derived
// from `E`.
//
// When `E` feeds any consumer besides `R2` the fusion works on a clone of `E`
// and leaves the original in place: the fused copy computes each tile against
// the *running* accumulator, which only the online correction on `R2` accounts
// for, so no other consumer may read it. That covers both softmax's normalizing
// divide (which needs `E` evaluated at the final accumulator) and a second
// consumer reduction over the same axis -- as in an attention chain, where one
// `exp` term is read by both a row sum and a `P @ V` contraction -- which is
// fused into the same loop by applying the fusion again. See
// `needsElementwiseClone`.
//
// This is exposed only as a pattern, via
// `populateDependantReductionFusionPatterns`. The intended entry point is
// `transform.structured.fuse_dependant_reduction_ops`, whose control
// function restricts the fusion to one explicitly named chain; there is no
// standalone pass, since fusing every chain a function happens to contain is
// not a useful default. The `test-linalg-dependant-reduction-fusion` test pass
// drives the pattern unconstrained so the patterns can be tested directly.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
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

/// Unit attribute that marks an `scf.for` as a tiled reduction loop. The
/// producer of the IR (e.g. `transform.structured.tile_using_for` followed by
/// `transform.annotate`) tags the loop with this attribute so the fusion
/// pattern can recognise an already-tiled producer reduction `R1`. A plain
/// `scf.for` carries no iterator-type metadata, so this marker is the only way
/// to tell that the loop iterates a reduction axis.
static constexpr StringLiteral kReductionLoopAttrName = "__reduction_loop__";

/// Resolve `val` through any chain of `tensor.extract_slice` ops to the
/// underlying source tensor. Inside a tiled reduction loop, the inner `R1`
/// generic reads `tensor.extract_slice` of the real input tensors; comparing
/// `R1`'s inputs against `E`'s requires looking through those tile slices.
static Value resolveSliceSource(Value val) {
  while (auto slice = val.getDefiningOp<tensor::ExtractSliceOp>())
    val = slice.getSource();
  return val;
}

/// Return true if `val` is statically known to be zero — either a constant
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

/// The `E` input operands that consume one of R1's results, with a parallel
/// list of which R1 result each one consumes. The legality check needs both:
/// the operands to verify their indexing maps, and the result index to recover
/// the inner reduction that produces each one.
struct R1AsElementwiseInputInfo {
  SmallVector<OpOperand *> operands;
  SmallVector<unsigned> r1ResultIdx;
};

/// Collect the `e` DPS inputs that consume a result of `r1`, where `r1` is the
/// `__reduction_loop__` `scf.for`: pre-fusion, the elementwise term `E`
/// consumes the loop's `iter_arg`-carried results, and each returned operand is
/// paired with the loop-result index it consumes.
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

/// All reduction `linalg.generic`s found in the body of a
/// `__reduction_loop__`-annotated `scf.for`, in program order. A loop may carry
/// several running accumulators (e.g. a fused softmax `(max, sum)` loop), each
/// produced by its own inner reduction generic.
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

/// Map each loop result of a `__reduction_loop__` `scf.for` back to the inner
/// reduction `linalg.generic` that produces its tile. The returned vector is
/// indexed by loop result number; an entry is null when the corresponding
/// result is not yielded directly from a reduction generic's tile (e.g. a
/// non-reduction passthrough result).
///
/// Each running accumulator is yielded as `tensor.insert_slice %red into %arg`,
/// where `%red` is the inner reduction's result and `%arg` is the carried
/// `iter_arg`. We therefore look at each `scf.yield` operand: if it is a
/// `tensor.insert_slice` whose source is defined by a reduction generic, that
/// generic produces this result.
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

/// Verify, for a single inner reduction `r1` of a multi-output reduction loop,
/// that it can be fused against the elementwise term `e`: `r1` must reduce over
/// exactly one innermost loop, every `r1` input (resolved *through* its tile
/// `extract_slice` to the source tensor) must also appear as an `e` input —
/// *except* inputs that are results of a sibling inner reduction
/// (`innerResults`), which are themselves broadcast running accumulators and
/// need not appear in `e` — and the shared inputs' indexing maps must agree
/// under a consistent injective dim-mapping φ from `r1`'s loop dims to `e`'s
/// loop dims, with φ total over `r1`'s loops and aligning `r1`'s reduction dim
/// with the `e` dim that carries R2's reduction axis (`eTiledDim`).
///
/// Note the comparison is against `E`, not `R2`: with the elementwise term left
/// unfused, `E` is the op that shares `R1`'s data inputs (e.g. `x` in
/// `exp(x - m)`), while `R2` reduces `E`'s result (possibly alongside other
/// contraction operands) and need not share any input with `R1`.
static LogicalResult checkInnerReductionAgainstElementwise(
    linalg::GenericOp r1, linalg::GenericOp e, unsigned eTiledDim,
    const llvm::SmallDenseSet<Value, 4> &innerResults) {
  SmallVector<unsigned> r1RedDims;
  r1.getReductionDims(r1RedDims);
  if (r1RedDims.size() != 1) {
    LLVM_DEBUG(
        DBGS() << "checkInnerReductionAgainstElementwise: failed — inner "
                  "R1 does not have exactly one reduction iterator ("
               << r1RedDims.size() << ").\n");
    return failure();
  }
  if (r1RedDims.front() != r1.getNumLoops() - 1) {
    LLVM_DEBUG(DBGS() << "checkInnerReductionAgainstElementwise: failed — "
                         "reduction iterator is not the innermost loop in "
                         "inner R1.\n");
    return failure();
  }

  // Every input of R1 must also appear as an input of E (besides R1's result
  // and any sibling reduction's result), and the two ops' indexing maps for
  // each shared input must agree up to a consistent injective mapping φ from
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
      LLVM_DEBUG(DBGS() << "checkInnerReductionAgainstElementwise: failed — R1 "
                           "input is not also an input of E: "
                        << in1->get() << "\n");
      return failure();
    }
    AffineMap m1 = r1.getMatchingIndexingMap(in1);
    AffineMap mE = e.getMatchingIndexingMap(inE);
    if (m1.getNumResults() != mE.getNumResults()) {
      LLVM_DEBUG(
          DBGS() << "checkInnerReductionAgainstElementwise: failed — shared "
                    "input has maps of different rank in R1 vs E (R1: "
                 << m1 << ", E: " << mE << ").\n");
      return failure();
    }
    for (auto [e1, e2] : llvm::zip_equal(m1.getResults(), mE.getResults())) {
      auto d1 = dyn_cast<AffineDimExpr>(e1);
      auto d2 = dyn_cast<AffineDimExpr>(e2);
      if (!d1 || !d2) {
        LLVM_DEBUG(DBGS() << "checkInnerReductionAgainstElementwise: failed — "
                             "shared input map has a non-dim affine expr (R1: "
                          << m1 << ", E: " << mE << ").\n");
        return failure();
      }
      if (!tryAddMapping(d1.getPosition(), d2.getPosition())) {
        LLVM_DEBUG(DBGS() << "checkInnerReductionAgainstElementwise: failed — "
                             "inconsistent dim mapping between R1 and E "
                             "derived from shared inputs (R1.d"
                          << d1.getPosition() << " -> {E.d"
                          << phi[d1.getPosition()] << ", E.d"
                          << d2.getPosition() << "}).\n");
        return failure();
      }
    }
  }
  // φ must be total over R1's loop dims so we can translate R1's init map
  // (and any other R1 map) into E's iter space during fusion.
  if (phi.size() != r1.getNumLoops()) {
    LLVM_DEBUG(
        DBGS() << "checkInnerReductionAgainstElementwise: failed — derived dim "
                  "mapping does not cover all of R1's loop dims (covered "
               << phi.size() << " of " << r1.getNumLoops() << ").\n");
    return failure();
  }
  auto redIt = phi.find(r1RedDims.front());
  if (redIt == phi.end() || redIt->second != eTiledDim) {
    LLVM_DEBUG(
        DBGS() << "checkInnerReductionAgainstElementwise: failed — R1's "
                  "reduction dim is not aligned with the E dim carrying "
                  "R2's reduction axis under the derived dim mapping.\n");
    return failure();
  }
  return success();
}

/// Find the `e` loop dim that carries `r2`'s reduction axis, where `r2`'s
/// single input `r2EOperand` reads `e`'s result.
///
/// `E`'s output map and `R2`'s input map both describe the same tensor (E's
/// result), so they can be aligned position-by-position: at tensor dim `i`,
/// `R2`'s map yields an `R2` loop dim and `E`'s output map yields an `E` loop
/// dim. The `E` dim sitting at the position where `R2` reads its reduction dim
/// is the axis along which `E` must be re-sliced when it is fused into R1's
/// tiled loop. Returns `nullopt` if either map is not a pure dim projection or
/// `r2RedDim` does not appear in `R2`'s map for that operand.
static std::optional<unsigned>
findElementwiseDimForR2ReductionDim(linalg::GenericOp e, linalg::GenericOp r2,
                                    OpOperand *r2EOperand, unsigned r2RedDim) {
  AffineMap r2Map = r2.getMatchingIndexingMap(r2EOperand);
  AffineMap eOutMap = e.getMatchingIndexingMap(e.getDpsInitOperand(0));
  if (r2Map.getNumResults() != eOutMap.getNumResults()) {
    LLVM_DEBUG(DBGS() << "findElementwiseDimForR2ReductionDim: failed — R2's "
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
      LLVM_DEBUG(DBGS() << "findElementwiseDimForR2ReductionDim: failed — a "
                           "non-dim affine expr in R2's map for E's result or "
                           "in E's output map (R2: "
                        << r2Map << ", E out: " << eOutMap << ").\n");
      return std::nullopt;
    }
    if (r2Dim.getPosition() == r2RedDim)
      return eDim.getPosition();
  }
  LLVM_DEBUG(DBGS() << "findElementwiseDimForR2ReductionDim: failed — R2's "
                       "reduction dim d"
                    << r2RedDim
                    << " does not appear in its map for E's result: " << r2Map
                    << "\n");
  return std::nullopt;
}

/// Whether this fusion must work on a *clone* of `e` rather than on `e` itself,
/// i.e. whether `e`'s result feeds any consumer besides `r2`.
///
/// Fusing `E` moves it into the loop, where each tile is computed against the
/// *running* value of `R1`'s accumulator rather than its final one. That is
/// exactly what the online correction accounts for — but only for `R2`'s
/// accumulator. `E`'s own full-extent result, materialized as a loop result, is
/// therefore stale in every tile but the last, so no consumer outside the loop
/// may read it. Concretely, for softmax `E = exp(x - m)`, the tile written at
/// step `t` is `exp(x - m_t)` and differs from the final `exp(x - m)` by
/// `exp(m - m_t)`; a normalizing divide reading it off the loop normalizes
/// stale numerators and gets row sums greater than one.
///
/// So whenever `E` has another consumer, fuse a clone and leave the original in
/// place, where it still reads `R1`'s final result off the loop and recomputes
/// the term correctly. That covers both shapes this comes up in:
///   * an all-parallel consumer (softmax's normalizing divide), which needs the
///     final-`R1` values;
///   * another consumer *reduction* over the same axis — an attention chain has
///     both a row sum and a `P @ V` contraction over one `exp` term — which is
///     a candidate `R2` in its own right and needs its own elementwise term to
///     move into the loop when it is fused in turn.
/// The cloned full-extent result is dead on arrival; `--remove-dead-values`
/// unwinds it along with the loop result carrying it.
static bool needsElementwiseClone(linalg::GenericOp e, linalg::GenericOp r2) {
  return !llvm::all_of(e.getResult(0).getUsers(), [&](Operation *user) {
    return user == r2.getOperation();
  });
}

/// How the tracked accumulators enter a value of `E`'s body. `Ind`/`Const`
/// describe values the accumulators do not reach at all; `Add`/`Mul` describe
/// the two separable shapes, writing `m` for the accumulators and `x` for the
/// data operands.
enum SeparabilityFact : uint8_t {
  /// `v` is independent of every block argument (a loop-invariant scalar).
  Const = 1 << 0,
  /// `v = h(x)`: no dependence on the accumulators.
  Ind = 1 << 1,
  /// `v = alpha(m) + h(x)`: the accumulators enter additively.
  Add = 1 << 2,
  /// `v = g(m) * h(x)`: the accumulators enter multiplicatively.
  Mul = 1 << 3,
};

/// Close a fact set under the implications between the facts: a value the
/// accumulators never reach trivially fits both separable shapes, with a
/// trivial accumulator part (`h(x) = 0 + h(x)` and `h(x) = 1 * h(x)`).
static uint8_t closeFacts(uint8_t facts) {
  if (facts & Const)
    facts |= Ind;
  if (facts & Ind)
    facts |= Add | Mul;
  return facts;
}

/// Verify that `E` is *multiplicatively separable* in the accumulators it
/// consumes, i.e. that its body computes
///
///   E(x, m) = f(x) * g(m)
///
/// where `m` are the values read through `accumulatorArgs` and `x` is
/// everything else. This is exactly the condition under which the online
/// correction is well defined: the correction factor is obtained by evaluating
/// `E` twice, at the new and the old accumulator, with a stand-in constant
/// substituted for the data operands, and separability is what makes `f` cancel
/// --
///
///   E(v, m_new) / E(v, m_old) = g(m_new) / g(m_old)  for every `v`
///
/// so a single scalar per parallel slice repairs `R2`'s accumulator. Without it
/// the ratio still evaluates to *something*, but that something depends on the
/// stand-in, and no single scalar is correct: for `E = (x - m)^2` with a tile
/// holding `x = 1` and `m` moving from `1` to `3`, the true sum is `4` while
/// any rescale of the stale accumulator `0` yields `0`.
///
/// The check is an abstract interpretation over the fact sets above, run as a
/// single forward pass -- `E`'s body is straight-line, with no control flow.
/// Each accumulator block argument is seeded `Add | Mul` (a bare `m` is both
/// `m + 0` and `m * 1`), other block arguments `Ind`, and values defined
/// outside the body `Const`. `E` is separable iff the yielded value holds
/// `Mul`.
///
/// The load-bearing rule is `exp`, which turns an additive dependence into a
/// multiplicative one -- `exp(alpha(m) + h(x)) = exp(alpha(m)) * exp(h(x))` --
/// and is why both shapes have to be tracked rather than just `Mul`. Softmax's
/// `exp(x - m)` reaches `Mul` through it: `x - m` is `Add` only, and the `exp`
/// converts it. A norm's `abs(x / m)` reaches `Mul` without it. Variance's
/// `(x - m)^2` is rejected: the `mulf` needs both operands `Mul`, and `x - m`
/// carries only `Add` with no `exp` to convert it.
///
/// Note this tracks *where the accumulators flow* rather than which opcodes
/// appear, so arithmetic confined to the data side stays invisible: the `mulf`
/// in a scaled `exp(qk * scale - m)` does not disturb the result.
static LogicalResult
checkElementwiseSeparability(linalg::GenericOp e,
                             ArrayRef<BlockArgument> accumulatorArgs) {
  Block *body = e.getBlock();
  llvm::SmallDenseMap<Value, uint8_t> facts;

  // Seed the block arguments. All tracked accumulators are seeded at once, so
  // the pass proves *joint* separability `E(x, m1, m2) = f(x) * g(m1, m2)` --
  // the correction substitutes new/old for all of them simultaneously.
  llvm::SmallDenseSet<Value, 4> accs(accumulatorArgs.begin(),
                                     accumulatorArgs.end());
  for (BlockArgument barg : body->getArguments())
    facts[barg] = closeFacts(accs.contains(barg) ? (Add | Mul) : Ind);

  // Values from an enclosing scope are invariant over `E`'s iteration space.
  auto factsOf = [&](Value v) -> uint8_t {
    auto it = facts.find(v);
    return it == facts.end() ? closeFacts(Const) : it->second;
  };
  auto has = [&](Value v, SeparabilityFact f) { return factsOf(v) & f; };

  for (Operation &op : body->without_terminator()) {
    // Only single-result scalar arithmetic is modelled; anything else (a
    // comparison, a select, a call) is opaque and leaves the result with no
    // facts, which rejects the chain unless the value is dead.
    if (op.getNumResults() != 1)
      continue;
    Value res = op.getResult(0);
    uint8_t f = 0;
    auto binary = [&](SeparabilityFact preserved) {
      Value lhs = op.getOperand(0), rhs = op.getOperand(1);
      if (has(lhs, Const) && has(rhs, Const))
        f |= Const;
      if (has(lhs, Ind) && has(rhs, Ind))
        f |= Ind;
      // `(a1 + a2)(m) + (h1 + h2)(x)` for the additive family, and
      // `(g1 g2)(m) * (h1 h2)(x)` for the multiplicative one.
      if (has(lhs, preserved) && has(rhs, preserved))
        f |= preserved;
    };
    // Propagate `facts` through a unary op that maps `from` to `to`, e.g. `exp`
    // maps `Add` to `Mul`. `Const`/`Ind` always survive a unary op.
    auto unary = [&](SeparabilityFact from, SeparabilityFact to) {
      Value arg = op.getOperand(0);
      f |= factsOf(arg) & (Const | Ind);
      if (has(arg, from))
        f |= to;
    };

    llvm::TypeSwitch<Operation *>(&op)
        .Case<arith::AddFOp, arith::SubFOp>([&](auto) { binary(Add); })
        .Case<arith::MulFOp, arith::DivFOp>([&](auto) { binary(Mul); })
        // `-(alpha + h) = (-alpha) + (-h)` and `-(g * h) = g * (-h)`.
        .Case<arith::NegFOp>([&](auto) { f = factsOf(op.getOperand(0)); })
        .Case<math::ExpOp, math::Exp2Op>([&](auto) { unary(Add, Mul); })
        // `log(g * h) = log(g) + log(h)`, the inverse bridge.
        .Case<math::LogOp, math::Log2Op>([&](auto) { unary(Mul, Add); })
        // `|g * h| = |g| * |h|`, and likewise for the (r)sqrt of a product.
        .Case<math::AbsFOp, math::SqrtOp, math::RsqrtOp>(
            [&](auto) { unary(Mul, Mul); })
        .Case<math::PowFOp>([&](auto) {
          Value base = op.getOperand(0), exp = op.getOperand(1);
          f |= factsOf(base) & factsOf(exp) & (Const | Ind);
          // `(g * h)^p = g^p * h^p` needs a *fixed* `p`: were the exponent to
          // vary with the data, `g(m)^p(x)` would still depend on `x`.
          if (has(base, Mul) && has(exp, Const))
            f |= Mul;
          // `c^(alpha + h) = c^alpha * c^h`.
          if (has(base, Const) && has(exp, Add))
            f |= Mul;
        })
        .Default([](Operation *) {});

    facts[res] = closeFacts(f);
  }

  auto yieldOp = cast<linalg::YieldOp>(body->getTerminator());
  if (yieldOp.getNumOperands() != 1) {
    LLVM_DEBUG(DBGS() << "checkElementwiseSeparability: failed -- E does not "
                         "yield exactly one value.\n");
    return failure();
  }
  Value term = yieldOp.getOperand(0);
  if (!has(term, Mul)) {
    LLVM_DEBUG(DBGS() << "checkElementwiseSeparability: failed -- E is not "
                         "multiplicatively separable in the accumulators it "
                         "consumes, so no per-slice scalar can correct R2's "
                         "running accumulator. Yielded value: "
                      << term << "\n");
    return failure();
  }
  // A yield that never reads an accumulator would make the correction the
  // constant 1; the chain is then not a dependent reduction at all.
  if (has(term, Ind)) {
    LLVM_DEBUG(DBGS() << "checkElementwiseSeparability: failed -- E does not "
                         "depend on any consumed accumulator.\n");
    return failure();
  }
  return success();
}

/// Verify that the `R1 -> E -> R2` chain can be fused into the already-tiled
/// producer loop `r1Loop`.
///
/// `r1Loop` is the already-tiled producer reduction: an `scf.for` carrying one
/// or more running accumulators as `iter_arg`s, whose loop results the
/// elementwise term `e` consumes. `resultToInner` maps each loop result index
/// to the inner per-tile reduction `linalg.generic` that produces it (null for
/// non-reduction results), as built by `mapLoopResultsToInnerReductions`. Each
/// inner reduction reduces only one tile and reads `tensor.extract_slice`s of
/// the real inputs. The loop's `step` is the tile size and `ub - lb` is the
/// full reduction extent.
///
/// The chain is `R1 -> E -> R2` with the elementwise term UNFUSED:
///   * `E` is all-parallel, consumes one or more R1 loop results plus the data
///     inputs it shares with `R1`, and produces a full-extent tensor.
///   * `R2` reduces `E`'s result. It may have several inputs (e.g. a GEMM-like
///     contraction), but exactly one of them is `E`'s result and *every* input
///     must be reduced along `R2`'s single reduction axis.
/// The per-inner-reduction conditions (reduction-dim alignment and the φ
/// shared-input mapping) are checked against `E` — not `R2` — for **every**
/// loop result that `E` consumes, because `E` is the op that shares `R1`'s data
/// inputs. Any other user of an R1 result must post-dominate `E`, and any other
/// user of `E`'s result must post-dominate `R2`.
/// On success, `eTiledDim` receives the `E` loop dim carrying R2's reduction
/// axis, which the retiling step needs.
static LogicalResult checkLegalFusionTriple(
    scf::ForOp r1Loop, ArrayRef<linalg::GenericOp> resultToInner,
    linalg::GenericOp e, linalg::GenericOp r2, unsigned &eTiledDim) {

  LLVM_DEBUG(
      DBGS() << "checkLegalFusionTriple: checking candidate:\n  R1 loop: "
             << *r1Loop << "\n  E: " << *e << "\n  R2: " << *r2 << "\n");
  if (r2->getNumResults() != 1 || r2.getNumDpsInits() != 1) {
    LLVM_DEBUG(DBGS() << "checkLegalFusionTriple: failed — R2 does not have "
                         "exactly one result/init (results: "
                      << r2->getNumResults()
                      << ", inits: " << r2.getNumDpsInits() << ").\n");
    return failure();
  }

  // (E1) E must be all-parallel with exactly one result/init: it is the
  // elementwise term feeding R2, not a reduction of its own.
  if (e->getNumResults() != 1 || e.getNumDpsInits() != 1) {
    LLVM_DEBUG(DBGS() << "checkLegalFusionTriple: failed — E does not have "
                         "exactly one result/init (results: "
                      << e->getNumResults() << ", inits: " << e.getNumDpsInits()
                      << ").\n");
    return failure();
  }
  if (e.getNumReductionLoops() != 0) {
    LLVM_DEBUG(
        DBGS() << "checkLegalFusionTriple: failed — E is not all-parallel "
                  "(it has "
               << e.getNumReductionLoops() << " reduction loops).\n");
    return failure();
  }

  // (E2) R2 may have MULTIPLE inputs (e.g. a GEMM-like contraction), but
  // exactly one of them must be E's result, and *every* input must be reduced
  // along the shared reduction axis — see (E2b) below.
  if (e->getBlock() != r2->getBlock() || e->getBlock() != r1Loop->getBlock()) {
    LLVM_DEBUG(
        DBGS() << "checkLegalFusionTriple: failed — R1 loop, E and R2 are "
                  "not all in the same block.\n");
    return failure();
  }
  OpOperand *r2EOperand = nullptr;
  for (OpOperand *in : r2.getDpsInputOperands()) {
    if (in->get() != e.getResult(0))
      continue;
    if (r2EOperand) {
      LLVM_DEBUG(DBGS() << "checkLegalFusionTriple: failed — R2 consumes E's "
                           "result more than once.\n");
      return failure();
    }
    r2EOperand = in;
  }
  if (!r2EOperand) {
    LLVM_DEBUG(DBGS() << "checkLegalFusionTriple: failed — no R2 input is E's "
                         "result.\n");
    return failure();
  }

  // R2 must have exactly one reduction loop, and it must be its innermost loop.
  SmallVector<unsigned> r2RedDims;
  r2.getReductionDims(r2RedDims);
  if (r2RedDims.size() != 1) {
    LLVM_DEBUG(DBGS() << "checkLegalFusionTriple: failed — R2 does not have "
                         "exactly one reduction iterator ("
                      << r2RedDims.size() << ").\n");
    return failure();
  }
  if (r2RedDims.front() != r2.getNumLoops() - 1) {
    LLVM_DEBUG(
        DBGS() << "checkLegalFusionTriple: failed — reduction iterator is "
                  "not the innermost loop in R2.\n");
    return failure();
  }

  // (E2b) Every R2 input must be reduced along R2's reduction axis, i.e. its
  // indexing map must reference `r2RedDim`. An input that does *not* is
  // broadcast across the reduction (a per-parallel-slice value); such an
  // operand would have to be re-derived per tile rather than simply re-sliced,
  // which the rewrite does not model. Requiring all inputs to carry the axis is
  // what makes "re-slice every R2 input to the current tile" a complete
  // description of the transform.
  for (OpOperand *in : r2.getDpsInputOperands()) {
    AffineMap m = r2.getMatchingIndexingMap(in);
    bool carriesRedDim = false;
    for (AffineExpr expr : m.getResults()) {
      auto dimExpr = dyn_cast<AffineDimExpr>(expr);
      if (!dimExpr) {
        LLVM_DEBUG(
            DBGS() << "checkLegalFusionTriple: failed — R2 input indexing "
                      "map has a non-dim affine expr: "
                   << m << "\n");
        return failure();
      }
      if (dimExpr.getPosition() == r2RedDims.front())
        carriesRedDim = true;
    }
    if (!carriesRedDim) {
      LLVM_DEBUG(DBGS() << "checkLegalFusionTriple: failed — R2 input is not "
                           "reduced along R2's reduction axis (map does not "
                           "reference dim "
                        << r2RedDims.front() << "): " << m << "\n");
      return failure();
    }
  }

  // (E3) Locate the E loop dim that carries R2's reduction axis. This is the
  // axis along which E must be re-sliced when it is cloned into R1's tiled
  // loop, and the axis R1's reduction dim must align with.
  std::optional<unsigned> eRedDim =
      findElementwiseDimForR2ReductionDim(e, r2, r2EOperand, r2RedDims.front());
  if (!eRedDim)
    return failure();
  eTiledDim = *eRedDim;

  // The producer reduction loop must have static, constant bounds: the tile
  // size is the loop `step` and the full reduction extent is `ub - lb`. That
  // extent must equal R2's (static) reduction extent, and the tile size must
  // evenly divide it.
  std::optional<int64_t> lb = getConstantIntValue(r1Loop.getLowerBound());
  std::optional<int64_t> ub = getConstantIntValue(r1Loop.getUpperBound());
  std::optional<int64_t> step = getConstantIntValue(r1Loop.getStep());
  if (!lb || !ub || !step || *step <= 0) {
    LLVM_DEBUG(
        DBGS() << "checkLegalFusionTriple: failed — R1 reduction loop does "
                  "not have constant, positive bounds/step.\n");
    return failure();
  }
  int64_t fullExtent = *ub - *lb;
  int64_t tileSize = *step;
  SmallVector<int64_t> ranges2 = r2.getStaticLoopRanges();
  int64_t r2RedRange = ranges2[r2RedDims.front()];
  if (ShapedType::isDynamic(r2RedRange)) {
    LLVM_DEBUG(
        DBGS() << "checkLegalFusionTriple: failed — R2 reduction range is "
                  "dynamic; fusion requires a static reduction "
                  "extent.\n");
    return failure();
  }
  if (r2RedRange != fullExtent) {
    LLVM_DEBUG(
        DBGS() << "checkLegalFusionTriple: failed — R2's reduction extent ("
               << r2RedRange << ") differs from the R1 loop extent ("
               << fullExtent << ").\n");
    return failure();
  }
  if (fullExtent % tileSize != 0) {
    LLVM_DEBUG(DBGS() << "checkLegalFusionTriple: failed — tile size "
                      << tileSize
                      << " does not evenly divide the reduction extent "
                      << fullExtent << ".\n");
    return failure();
  }

  // (E4) E's extent along the axis carrying R2's reduction must match the R1
  // loop extent too, so that re-slicing E to the tile is well defined.
  SmallVector<int64_t> rangesE = e.getStaticLoopRanges();
  int64_t eRedRange = rangesE[*eRedDim];
  if (ShapedType::isDynamic(eRedRange) || eRedRange != fullExtent) {
    LLVM_DEBUG(
        DBGS() << "checkLegalFusionTriple: failed — E's extent along the "
                  "axis carrying R2's reduction ("
               << eRedRange << ") differs from the R1 loop extent ("
               << fullExtent << ").\n");
    return failure();
  }

  // The E operands that consume a loop result. There must be at least one:
  // this is the R1 -> E data dependence that makes the chain fusable.
  R1AsElementwiseInputInfo r1AsE = collectR1AsElementwiseInputs(r1Loop, e);
  if (r1AsE.operands.empty()) {
    LLVM_DEBUG(DBGS() << "checkLegalFusionTriple: failed — E does not consume "
                         "any result of the R1 loop.\n");
    return failure();
  }

  // (E5) `E` must be multiplicatively separable in the accumulators it reads,
  // or the online correction it is asked to derive does not exist.
  SmallVector<BlockArgument> accumulatorArgs =
      llvm::map_to_vector(r1AsE.operands, [&](OpOperand *use) {
        return e.getMatchingBlockArgument(use);
      });
  if (failed(checkElementwiseSeparability(e, accumulatorArgs)))
    return failure();

  // The set of all inner reduction results in the loop, used to skip
  // sibling-reduction inputs in the per-inner-reduction φ check.
  llvm::SmallDenseSet<Value, 4> innerResults;
  for (linalg::GenericOp inner : resultToInner)
    if (inner)
      innerResults.insert(inner.getResult(0));

  // Each R1 result consumed by E must be broadcast across the E axis that
  // carries R2's reduction — i.e. the indexing map for that operand must not
  // reference `eRedDim`. Conservatively also require a pure dim-expr
  // projection. This is what makes the running accumulator a per-parallel-slice
  // scalar that the online correction can rescale.
  for (OpOperand *r1AsEInput : r1AsE.operands) {
    AffineMap r1AsEMap = e.getMatchingIndexingMap(r1AsEInput);
    for (AffineExpr expr : r1AsEMap.getResults()) {
      auto dimExpr = dyn_cast<AffineDimExpr>(expr);
      if (!dimExpr) {
        LLVM_DEBUG(DBGS() << "checkLegalFusionTriple: failed — R1-as-E-input "
                             "indexing map has a non-dim affine expr: "
                          << r1AsEMap << "\n");
        return failure();
      }
      if (dimExpr.getPosition() == *eRedDim) {
        LLVM_DEBUG(
            DBGS() << "checkLegalFusionTriple: failed — R1's result is not "
                      "broadcast across the E axis carrying R2's reduction "
                      "(map references dim "
                   << dimExpr.getPosition() << "): " << r1AsEMap << "\n");
        return failure();
      }
    }
  }

  // For each consumed loop result, recover the inner reduction that produces it
  // and verify the reduction-dim alignment and shared-input φ mapping against
  // E. If any consumed result fails, the whole fusion is rejected.
  for (auto [operand, resultIdx] :
       llvm::zip_equal(r1AsE.operands, r1AsE.r1ResultIdx)) {
    linalg::GenericOp inner = resultToInner[resultIdx];
    if (!inner) {
      LLVM_DEBUG(
          DBGS() << "checkLegalFusionTriple: failed — consumed loop result "
                 << resultIdx
                 << " is not produced by an inner reduction generic.\n");
      return failure();
    }
    if (failed(checkInnerReductionAgainstElementwise(inner, e, *eRedDim,
                                                     innerResults)))
      return failure();
  }

  // (5a) R2's region must be a single-combiner sum reduction.
  SmallVector<Operation *> combiners;
  if (!matchReduction(r2.getRegionOutputArgs(), /*redPos=*/0, combiners)) {
    LLVM_DEBUG(
        DBGS() << "checkLegalFusionTriple: failed — R2's region does not "
                  "match a reduction pattern.\n");
    return failure();
  }
  if (combiners.size() != 1) {
    LLVM_DEBUG(DBGS() << "checkLegalFusionTriple: failed — R2's reduction has "
                      << combiners.size()
                      << " combiners, expected exactly 1.\n");
    return failure();
  }
  if (!isa<arith::AddFOp>(combiners.front())) {
    LLVM_DEBUG(
        DBGS() << "checkLegalFusionTriple: failed — R2's combiner is not "
                  "arith.addf: "
               << *combiners.front() << "\n");
    return failure();
  }

  // (5b) R2's init must be the additive identity (zero), produced directly or
  // through a linalg.fill of zero into an empty tensor.
  Value r2Init = r2.getDpsInitOperand(0)->get();
  if (!isDefinedAsZero(r2Init)) {
    LLVM_DEBUG(
        DBGS() << "checkLegalFusionTriple: failed — R2's init is not the "
                  "additive identity (zero): "
               << r2Init << "\n");
    return failure();
  }

  // (5c) Restrict to supported floating-point element types.
  Type elt = getElementTypeOrSelf(r2->getResultTypes().front());
  if (!(elt.isF16() || elt.isBF16() || elt.isF32() || elt.isF64())) {
    LLVM_DEBUG(DBGS() << "checkLegalFusionTriple: failed — R2's element type "
                      << elt
                      << " is not a supported floating-point type "
                         "(f16/bf16/f32/f64).\n");
    return failure();
  }

  // (6a) Any other user of any R1 loop result (besides E) must post-dominate E
  // so that re-routing the values through the fused op is safe. Note the anchor
  // is E, not R2: E is the op that directly consumes the R1 loop results and
  // the one that gets cloned into the loop first.
  PostDominanceInfo postDom;
  for (Value r1Result : r1Loop->getResults()) {
    for (Operation *user : r1Result.getUsers()) {
      if (user == e.getOperation())
        continue;
      if (!postDom.postDominates(user, e.getOperation())) {
        LLVM_DEBUG(DBGS() << "checkLegalFusionTriple: failed — user of an R1 "
                             "result does not post-dominate E: "
                          << *user << "\n");
        return failure();
      }
    }
  }

  // Note there is no condition on the users of E's result: any user besides R2
  // makes the fusion work on a clone and leave the original E — and therefore
  // those users — exactly where they are. See `needsElementwiseClone`.
  return success();
}

/// Rewrite `offsets`/`sizes` so that every position whose indexing-map result
/// references `tiledDim` is cut to the current reduction tile: offset = loop
/// IV, size = `tileSize`. Positions already cut to `tileSize` are left alone.
/// Returns true if anything changed.
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

/// `tileAndFuseConsumer` tiles a fused consumer along the dims it shares with
/// the loop's yielded slices only. Since R1's result carries no reduction axis,
/// nothing gets restricted to the current tile: the fused ops still read (and,
/// for `E`, write) the *whole* reduction extent, e.g. `[64, 512]` instead of
/// one
/// `[64, 32]` tile.
///
/// Re-slice `fused` along `tiledDim` — the `fused` loop dim that carries the
/// reduction axis tiled by `redLoop`. Every operand whose indexing map
/// references `tiledDim` and is fed by a full-extent `tensor.extract_slice` is
/// cut to the tile. For an all-parallel `E` this includes its *destination*, so
/// the op's result type shrinks as well; the op is then rebuilt and the
/// `tensor.insert_slice` yielding it is re-sliced to match. For a reduction
/// consumer `R2` the output map never references `tiledDim`, so only inputs
/// change and this reduces to an in-place slice rewrite.
static FailureOr<linalg::GenericOp>
retileFusedConsumerToTile(RewriterBase &rewriter, linalg::GenericOp fused,
                          unsigned tiledDim, scf::ForOp redLoop,
                          int64_t tileSize) {
  Value iv = redLoop.getInductionVar();

  // Re-slice every operand (inputs *and* inits) fed by a full-extent slice.
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

/// A running accumulator of the fused loop, as seen by the fused `E`: the `E`
/// input operand reading the accumulator's *new* (current-tile) value, paired
/// with its *old* (previous running) value.
struct ConsumedAccumulator {
  OpOperand *use; // fused E input operand; use->get() is the new value.
  Value oldValue; // previous running value (the inner reduction's DPS init).
};

/// Returns the constant that eliminates one operand's contribution to `op`,
/// i.e. the value `c` such that the op becomes the identity on its *other*
/// operand: `0` for the additive family (`addf`/`subf`/`addi`/`subi`) and `1`
/// for the multiplicative family (`mulf`/`divf`/`muli`). Unlike
/// `arith::getNeutralElement`, this is defined for the non-commutative ops
/// (`subf`, `divf`) we encounter in correction terms. Returns `nullopt` for op
/// kinds without such a constant.
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

/// Emit, at the current insertion point of `b`, `E`'s body evaluated for the
/// online correction: `E` isolated on its accumulator inputs, with every other
/// (data) input neutralized at its use point.
///
/// `uses` binds each accumulator-reading `E` input operand to the scalar to
/// substitute for it (the current-tile value for the "new" factor, the previous
/// running value for the "old" one).
///
/// Because `E` is a separate op we can clone its body directly — no backward
/// slice is needed, `E`'s yielded value *is* the term. Data block arguments are
/// replaced by the constant that eliminates their contribution to the consuming
/// op (`getOperandEliminatingConstant`): `1.0` for `mulf`/`divf`, `0.0` for
/// `addf`/`subf`, decided per use so a data op drops out cleanly instead of
/// contributing a spurious magnitude. This is sound because the data inputs
/// cancel in the new/old ratio — for softmax `E = exp(x - m)` gives
/// `exp(x - m_new)/exp(x - m_old) = exp(m_old - m_new)` for any `x`.
/// Returns the cloned term value in `b`'s region.
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

/// Build the elementwise op computing the online rescale factor for the current
/// tile: `term(new) / term(old)`, where `term` is `E` isolated on the running
/// accumulators it consumes (see `emitCorrectionTerm`).
///
/// The two `term` evaluations and the division are `linalg.generic`s over the
/// accumulator's iteration space, so the factor has `r2`'s output shape. Each
/// accumulator is read through its own `E` indexing map with `eTiledDim`
/// projected out — legality guarantees those maps do not reference it, so the
/// projection is lossless. Returns null on failure.
static Value calculateCorrectionFactor(PatternRewriter &rewriter,
                                       linalg::GenericOp e,
                                       linalg::GenericOp r2, unsigned eTiledDim,
                                       ArrayRef<ConsumedAccumulator> accs) {
  Location loc = r2.getLoc();

  // The factor is built over `r2`'s *parallel* iteration space so it has `r2`'s
  // accumulator shape — for a contraction that space is wider than `E`'s (e.g.
  // GEMM's extra N dim), and the accumulator must broadcast over it. Translate
  // each accumulator's map from `E`'s dims into `r2`'s by aligning `E`'s output
  // map with `r2`'s map for `E`'s result position-by-position (the same
  // correspondence `findElementwiseDimForR2ReductionDim` uses for one dim).
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
  // Unmapped E dims cannot appear in an accumulator map (legality restricts
  // those to dims shared with R2); use a placeholder to surface a violation as
  // a failed projection rather than silently miscompiling.
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

  // The factor takes R2's accumulator shape, with an identity map over its
  // rank.
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

/// Rescale the fused `R2`'s running accumulator by the per-tile online factor,
/// so the partial sums carried from earlier tiles stay valid when `R1`'s
/// accumulator changes.
///
/// The accumulators are read by the fused `E` (`fusedE`), whose inputs hold
/// each running accumulator's *new* (current-tile) value; the matching *old*
/// value is the destination of the `tensor.insert_slice` that yields it from
/// the loop, i.e. the loop's `iter_arg`. `fusedR2`'s DPS init is replaced by
/// `init * term(new)/term(old)`.
static LogicalResult correctFusedR2Accumulator(PatternRewriter &rewriter,
                                               linalg::GenericOp fusedE,
                                               linalg::GenericOp fusedR2,
                                               unsigned eTiledDim) {
  auto loop = fusedR2->getParentOfType<scf::ForOp>();
  assert(loop && "expected fusedR2 to live inside the reduction loop");

  // Current-tile value -> previous running value. The current-tile value is an
  // inner reduction's result (yielded via `insert_slice %new into %iterArg`);
  // its previous value is that reduction's own DPS init, i.e. the tile slice of
  // the `iter_arg` it accumulates into. Using the init rather than the raw
  // `iter_arg` keeps the old value readable *after* the inner reduction has
  // run: the init is an immutable SSA value, so bufferization copies the
  // accumulator instead of letting the reduction write in place.
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
    LLVM_DEBUG(DBGS() << "correctFusedR2Accumulator: failed — fused E does not "
                         "consume a running accumulator of the loop.\n");
    return failure();
  }

  rewriter.setInsertionPoint(fusedR2);
  Value factor =
      calculateCorrectionFactor(rewriter, fusedE, fusedR2, eTiledDim, accs);
  if (!factor) {
    LLVM_DEBUG(DBGS() << "correctFusedR2Accumulator: failed — could not build "
                         "the correction factor from E.\n");
    return failure();
  }

  // Scale the running accumulator by the factor; accumulator, factor and result
  // share a shape, so default identity maps suffice.
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

/// Whether `value` transitively depends on `op`, i.e. `op` is in its backward
/// slice. Used to tell the operands that may be hoisted above the reduction
/// loop from the ones that are computed *by* it.
static bool dependsOnOp(Value value, Operation *op) {
  if (value.getDefiningOp() == op)
    return true;
  SetVector<Operation *> slice;
  if (failed(getBackwardSlice(value, &slice)))
    // Be conservative: an unknown slice may well contain `op`.
    return true;
  return slice.contains(op);
}

/// Hoist the definitions of `consumer`'s operands above `insertionPoint` (the
/// reduction loop), for those that are not already defined before it.
///
/// `scf::tileAndFuseConsumer` moves the loop nest up to just before its first
/// user, which for our chain is the consumer being fused; that move is rejected
/// by its dominance check when any consumer operand is defined *after* the
/// loop. Tiling routinely leaves such operands behind: `E`'s and `R2`'s
/// destinations (a `tensor.empty` / `linalg.fill`) and, when an enclosing
/// `scf.forall` tiled a parallel dim, the `tensor.extract_slice` feeding a data
/// input — e.g. the per-batch slice of `V` in an attention chain, which the
/// outer tiling emits just before the contraction and therefore below the loop.
///
/// Operands that depend on the loop (`R1`'s results, and `E`'s result when
/// hoisting for `R2`) are skipped: moving their definitions would try to move
/// the loop above itself.
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
    LLVM_DEBUG(DBGS() << "hoistConsumerOperandsBefore: failed — could not move "
                         "operand definitions of "
                      << *consumer.getOperation() << " before the loop.\n");
    return failure();
  }
  LLVM_DEBUG(DBGS() << "hoistConsumerOperandsBefore: hoisted " << toHoist.size()
                    << " operand definition(s) above the loop.\n");
  return success();
}

/// Fuse the elementwise term `e` and then the consumer reduction `r2` into the
/// already-tiled producer reduction loop `r1Loop`.
///
/// Both are fused with `scf::tileAndFuseConsumer`, which finds the loop results
/// each consumer reads, locates the `tensor.insert_slice`s that yield them, and
/// clones the consumer inside the loop computing the corresponding tile of its
/// result in place. `E` is fused first so that its result becomes a loop result
/// that `R2` can then be fused against. Both consumers' operand definitions are
/// hoisted above the loop beforehand (see `hoistConsumerOperandsBefore`), and
/// each fused clone
/// is then re-sliced to the reduction tile (see `retileFusedConsumerToTile`):
/// `eTiledDim` is `E`'s tiled dim, `R2`'s is its own reduction dim. Finally the
/// online correction rescales `R2`'s running accumulator (see
/// `correctFusedR2Accumulator`).
static FailureOr<scf::SCFFuseConsumerOfSliceResult>
fuseElementwiseAndR2IntoTiledR1Loop(PatternRewriter &rewriter,
                                    linalg::GenericOp e, linalg::GenericOp r2,
                                    scf::ForOp r1Loop, unsigned eTiledDim,
                                    int64_t tileSize) {
  // Fuse a clone of `E` instead of `E` itself when another consumer reduction
  // needs the term too (see `needsElementwiseClone`), leaving the original for
  // those consumers. The clone is inserted immediately *before* `E`, which
  // keeps it above every other user of an `R1` loop result — including the
  // original `E` — so the loop can still be moved to just before the op being
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

  // Hoist BOTH consumers' operand definitions above the loop up front, so a
  // consumer that cannot be made to dominate the loop is rejected before any
  // fusion has run. Doing it lazily per consumer would leave `E` fused into the
  // loop when `R2` turns out not to be hoistable.
  for (linalg::GenericOp consumer : {e, r2})
    if (failed(hoistConsumerOperandsBefore(rewriter, consumer,
                                           r1Loop.getOperation())))
      return failure();

  SmallVector<LoopLikeOpInterface> loops = {
      cast<LoopLikeOpInterface>(r1Loop.getOperation())};

  SmallVector<unsigned> r2RedDims;
  r2.getReductionDims(r2RedDims);
  assert(r2RedDims.size() == 1 && "legality guarantees a single R2 reduction");

  // Fuse E, then R2, each into the (progressively rewritten) loop nest.
  // `tileAndFuseConsumer` updates `loops` in place with the new loop ops.
  FailureOr<scf::SCFFuseConsumerOfSliceResult> fuseResult;
  SmallVector<linalg::GenericOp> fusedOps; // [0] = fused E, [1] = fused R2.
  for (auto [consumer, tiledDim] :
       {std::pair{e, eTiledDim}, std::pair{r2, r2RedDims.front()}}) {
    rewriter.setInsertionPoint(consumer);
    fuseResult = scf::tileAndFuseConsumer(rewriter, consumer.getOperation(),
                                          loops, /*fn=*/nullptr);
    if (failed(fuseResult)) {
      LLVM_DEBUG(DBGS() << "fuseElementwiseAndR2IntoTiledR1Loop: failed — "
                           "tileAndFuseConsumer did not succeed for "
                        << *consumer.getOperation() << "\n");
      return failure();
    }
    if (fuseResult->tiledOps.empty()) {
      LLVM_DEBUG(DBGS() << "fuseElementwiseAndR2IntoTiledR1Loop: failed — "
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
          DBGS() << "fuseElementwiseAndR2IntoTiledR1Loop: failed — fused "
                    "op is not a linalg.generic.\n");
      return failure();
    }
    auto redLoop = cast<scf::ForOp>(loops.front().getOperation());
    FailureOr<linalg::GenericOp> retiled =
        retileFusedConsumerToTile(rewriter, fused, tiledDim, redLoop, tileSize);
    if (failed(retiled))
      return failure();
    // Retiling may rebuild the op (E's result type shrinks), so track the
    // current one: the correction reads the fused E's accumulator operands.
    fusedOps.push_back(*retiled);

    LLVM_DEBUG(
        DBGS() << "=== after re-slicing the fused " << consumer->getName()
               << " to the reduction tile ===\n"
               << *redLoop->getParentWithTrait<OpTrait::IsIsolatedFromAbove>()
               << "\n");
  }

  // Rescale R2's running accumulator by the per-tile online factor derived from
  // the fused E.
  if (failed(correctFusedR2Accumulator(rewriter, fusedOps[0], fusedOps[1],
                                       eTiledDim)))
    return failure();

  LLVM_DEBUG(DBGS() << "=== after applying the online correction ===\n"
                    << *loops.front()
                            ->getParentWithTrait<OpTrait::IsIsolatedFromAbove>()
                    << "\n");

  return fuseResult;
}

/// Pattern that matches the consumer reduction `R2` of an unfused
/// `R1 -> E -> R2` chain, walks back through the `R2` input produced by the
/// elementwise term `E`, and from `E`'s inputs to an *already-tiled* producer
/// reduction `R1` (an `scf.for` marked with `__reduction_loop__`). If the
/// triple satisfies the legality conditions it fuses the chain into the
/// producer's tiled loop. The tile size is recovered from the producer loop's
/// `step` (this pass no longer tiles `R1` itself).
///
/// The chain is matched from the bottom up:
///   `R2` (reduction; all inputs reduced along its reduction axis)
///     -> one of its inputs is `E`'s result
///        -> one of `E`'s inputs is a result of the `__reduction_loop__` loop.
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
      LLVM_DEBUG(DBGS() << "FuseDependentReductionsPattern: failed — R2 has "
                           "no reduction loops: "
                        << *r2.getOperation() << "\n");
      return failure();
    }

    // R2 may have several inputs (e.g. a GEMM-like contraction); exactly one of
    // them must be produced by the elementwise term E (an all-parallel
    // `linalg.generic`). Scan for it, and reject an ambiguous match where more
    // than one input is produced by a distinct candidate E.
    linalg::GenericOp e;
    for (OpOperand *in : r2.getDpsInputOperands()) {
      auto candidateE = in->get().getDefiningOp<linalg::GenericOp>();
      if (!candidateE || candidateE.getNumReductionLoops() != 0)
        continue;
      if (!controlFn(in)) {
        LLVM_DEBUG(
            DBGS() << "FuseDependentReductionsPattern: skipping candidate "
                      "E — control function vetoed fusing R2 with it.\n");
        continue;
      }
      if (e && candidateE != e) {
        LLVM_DEBUG(
            DBGS() << "FuseDependentReductionsPattern: failed — more than "
                      "one R2 input is produced by an all-parallel "
                      "elementwise linalg.generic; the chain is "
                      "ambiguous.\n");
        return failure();
      }
      e = candidateE;
    }
    if (!e) {
      LLVM_DEBUG(
          DBGS() << "FuseDependentReductionsPattern: failed — no R2 input "
                    "is produced by an all-parallel elementwise "
                    "linalg.generic.\n");
      return failure();
    }
    LLVM_DEBUG(
        DBGS() << "FuseDependentReductionsPattern: found elementwise term "
                  "E: "
               << *e.getOperation() << "\n");

    // E may have any number of DPS inputs. Try each as a candidate producer R1:
    // it must be defined by an `scf.for` annotated with `__reduction_loop__`
    // (an already-tiled reduction) in the same block whose body holds one or
    // more reduction generics, and the triple must satisfy the legality checks.
    // Use the first candidate that succeeds.
    scf::ForOp r1Loop;
    unsigned eTiledDim = 0;
    for (OpOperand *in : e.getDpsInputOperands()) {
      auto candidateLoop = in->get().getDefiningOp<scf::ForOp>();
      if (!candidateLoop || !candidateLoop->hasAttr(kReductionLoopAttrName))
        continue;
      if (!controlFn(in)) {
        LLVM_DEBUG(DBGS() << "FuseDependentReductionsPattern: skipping "
                             "candidate R1 — control function vetoed fusing "
                             "through this operand.\n");
        continue;
      }
      if (candidateLoop->getBlock() != e->getBlock()) {
        LLVM_DEBUG(DBGS() << "FuseDependentReductionsPattern: skipping "
                             "candidate R1 — not in the same block as E.\n");
        continue;
      }
      if (collectInnerReductionGenerics(candidateLoop).empty()) {
        LLVM_DEBUG(DBGS() << "FuseDependentReductionsPattern: skipping "
                             "candidate R1 — loop body does not contain any "
                             "reduction linalg.generic.\n");
        continue;
      }
      SmallVector<linalg::GenericOp> resultToInner =
          mapLoopResultsToInnerReductions(candidateLoop);
      if (failed(checkLegalFusionTriple(candidateLoop, resultToInner, e, r2,
                                        eTiledDim))) {
        LLVM_DEBUG(DBGS() << "FuseDependentReductionsPattern: skipping "
                             "candidate R1 — checkLegalFusionTriple rejected "
                             "the candidate.\n");
        continue;
      }
      r1Loop = candidateLoop;
      break;
    }
    if (!r1Loop) {
      LLVM_DEBUG(DBGS() << "FuseDependentReductionsPattern: failed — no DPS "
                           "input of E is an already-tiled fusable producer "
                           "reduction: "
                        << *r2.getOperation() << "\n");
      return failure();
    }
    LLVM_DEBUG(DBGS() << "FuseDependentReductionsPattern: selected R1 loop: "
                      << *r1Loop.getOperation() << "\n");

    // The tile size is the producer reduction loop's step;
    // checkLegalFusionTriple has verified it is a constant, positive value.
    int64_t tileSize = *getConstantIntValue(r1Loop.getStep());

    // Clone E and R2 into the existing tiled loop, then cut both down to the
    // current reduction tile.
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
