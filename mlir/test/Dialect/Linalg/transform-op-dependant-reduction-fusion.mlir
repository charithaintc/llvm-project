// RUN: mlir-opt --transform-interpreter -split-input-file -verify-diagnostics %s | FileCheck %s

// Tests for `transform.structured.fuse_dependant_reduction_ops`, which
// fuses an `R1 -> E -> R2` chain into `R1`'s already-tiled loop. Unlike the
// `--test-linalg-dependant-reduction-fusion` test pass (see
// dependant-reduction-fusion.mlir), the transform op constrains the fusion to
// exactly the triple named by its handles.

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

// Softmax `R1 -> E -> R2`: `m = max_j x`, `p = exp(x - m)`, `s = sum_j p`.
// CHECK-LABEL: func.func @softmax
// CHECK-SAME:      %[[X:[a-zA-Z0-9_]+]]: tensor<64x512xf32>
func.func @softmax(%arg0: tensor<64x512xf32>) -> tensor<64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %c32 = arith.constant 32 : index
  %0 = tensor.empty() : tensor<64xf32>
  %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>

  // R1: `max` reduction, already tiled along the reduction axis.
  %2 = scf.for %arg1 = %c0 to %c512 step %c32 iter_args(%arg2 = %1) -> (tensor<64xf32>) {
    %es = tensor.extract_slice %arg0[0, %arg1] [64, 32] [1, 1] : tensor<64x512xf32> to tensor<64x32xf32>
    %es1 = tensor.extract_slice %arg2[0] [64] [1] : tensor<64xf32> to tensor<64xf32>
    %m = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%es : tensor<64x32xf32>) outs(%es1 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %mx = arith.maximumf %in, %out : f32
      linalg.yield %mx : f32
    } -> tensor<64xf32>
    %ins = tensor.insert_slice %m into %arg2[0] [64] [1] : tensor<64xf32> into tensor<64xf32>
    scf.yield %ins : tensor<64xf32>
  } {__reduction_loop__}

  // E: the elementwise term, left unfused from R2. Its `tensor.empty` init sits
  // below the loop, as tiling would leave it; the op hoists it.
  %3 = tensor.empty() : tensor<64x512xf32>
  %4 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %2 : tensor<64x512xf32>, tensor<64xf32>) outs(%3 : tensor<64x512xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %d = arith.subf %in, %in_1 : f32
    %e = math.exp %d : f32
    linalg.yield %e : f32
  } -> tensor<64x512xf32>

  // R2: `sum` reduction over E's result, its `linalg.fill` init likewise below.
  %5 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %6 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%4 : tensor<64x512xf32>) outs(%5 : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %a = arith.addf %in, %out : f32
    linalg.yield %a : f32
  } -> tensor<64xf32>
  return %6 : tensor<64xf32>
}

// Both inits are hoisted above the loop, which now carries three accumulators:
// the running max, E's full result, and the running sum.
// CHECK-DAG:     %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:     %[[NINF:.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:     %[[MINIT:.+]] = linalg.fill ins(%[[NINF]] : f32)
// CHECK-DAG:     %[[EINIT:.+]] = tensor.empty() : tensor<64x512xf32>
// CHECK-DAG:     %[[SINIT:.+]] = linalg.fill ins(%[[ZERO]] : f32)
// CHECK:         scf.for %[[IV:[a-zA-Z0-9_]+]] = %{{[a-zA-Z0-9_]+}} to %{{[a-zA-Z0-9_]+}} step %{{[a-zA-Z0-9_]+}}
// CHECK-SAME:        iter_args(%[[MARG:[a-zA-Z0-9_]+]] = %[[MINIT]], %[[EARG:[a-zA-Z0-9_]+]] = %[[EINIT]], %[[SARG:[a-zA-Z0-9_]+]] = %[[SINIT]])
// CHECK-SAME:        -> (tensor<64xf32>, tensor<64x512xf32>, tensor<64xf32>)

// R1 over this tile. `%[[MOLD]]` is its DPS init, i.e. the previous running max.
// CHECK:           %[[MOLD:.+]] = tensor.extract_slice %[[MARG]][0] [64] [1]
// CHECK:           %[[MNEW:.+]] = linalg.generic
// CHECK-SAME:          outs(%[[MOLD]] :
// CHECK:             arith.maximumf

// The fused E, re-sliced from the full 512 extent down to the 32-wide tile and
// reading the new max.
// CHECK:           %[[XT:.+]] = tensor.extract_slice %[[X]][0, %[[IV]]] [64, 32] [1, 1]
// CHECK:           %[[ET:.+]] = tensor.extract_slice %[[EARG]][0, %[[IV]]] [64, 32] [1, 1]
// CHECK:           %[[P:.+]] = linalg.generic
// CHECK-SAME:          ins(%[[XT]], %[[MNEW]] : tensor<64x32xf32>, tensor<64xf32>)
// CHECK-SAME:          outs(%[[ET]] : tensor<64x32xf32>)
// CHECK:             arith.subf
// CHECK:             math.exp

// The online correction: E isolated on the max input, evaluated at both the new
// and the old max, their ratio scaling the running sum.
// CHECK:           %[[SOLD:.+]] = tensor.extract_slice %[[SARG]][0] [64] [1]
// CHECK:           %[[TNEW:.+]] = linalg.generic {{.*}}ins(%[[MNEW]] :
// CHECK:             arith.subf %[[ZERO]],
// CHECK:             math.exp
// CHECK:           %[[TOLD:.+]] = linalg.generic {{.*}}ins(%[[MOLD]] :
// CHECK:             arith.subf %[[ZERO]],
// CHECK:             math.exp
// CHECK:           %[[F:.+]] = linalg.elementwise kind=#linalg.elementwise_kind<div> ins(%[[TNEW]], %[[TOLD]] : tensor<64xf32>, tensor<64xf32>)
// CHECK:           %[[SCALED:.+]] = linalg.elementwise kind=#linalg.elementwise_kind<mul> ins(%[[SOLD]], %[[F]] : tensor<64xf32>, tensor<64xf32>)

// The fused R2 accumulates this tile's terms into the rescaled running sum.
// CHECK:           %[[SNEW:.+]] = linalg.generic
// CHECK-SAME:          ins(%[[P]] : tensor<64x32xf32>)
// CHECK-SAME:          outs(%[[SCALED]] : tensor<64xf32>)
// CHECK:             arith.addf
// CHECK:           tensor.insert_slice %[[P]] into %[[EARG]][0, %[[IV]]] [64, 32] [1, 1]
// CHECK:           tensor.insert_slice %[[SNEW]] into %[[SARG]][0] [64] [1]

// The op returns a handle to the fused loop, which the schedule annotates.
// CHECK:         } {fused_reduction_loop}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %gen = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // #1 = max (R1, already tiled), #2 = exp term (E), #3 = sum (R2).
    %max_op, %e_op, %sum_op = transform.split_handle %gen
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %loop = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %fused = transform.structured.fuse_dependant_reduction_ops %e_op, %sum_op into %loop
        : (!transform.any_op, !transform.any_op, !transform.any_op) -> !transform.any_op
    transform.annotate %fused "fused_reduction_loop" : !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

// The full softmax: @softmax plus the normalizing divide `out = p / s`, which also
// reads `E`. `E` therefore has a consumer besides `R2`, so a *clone* is fused and
// the original stays outside. This is required for correctness: the fused copy's
// tiles use the running max and are off by `exp(m - m_tile)` in all but the last,
// which the correction only repairs for `s`.
// CHECK-LABEL: func.func @softmax_with_normalizing_divide
// CHECK-SAME:      %[[X:[a-zA-Z0-9_]+]]: tensor<64x512xf32>
func.func @softmax_with_normalizing_divide(%arg0: tensor<64x512xf32>) -> tensor<64x512xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %c32 = arith.constant 32 : index
  %0 = tensor.empty() : tensor<64xf32>
  %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>

  // R1: `max` reduction, already tiled along the reduction axis.
  %2 = scf.for %arg1 = %c0 to %c512 step %c32 iter_args(%arg2 = %1) -> (tensor<64xf32>) {
    %es = tensor.extract_slice %arg0[0, %arg1] [64, 32] [1, 1] : tensor<64x512xf32> to tensor<64x32xf32>
    %es1 = tensor.extract_slice %arg2[0] [64] [1] : tensor<64xf32> to tensor<64xf32>
    %m = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%es : tensor<64x32xf32>) outs(%es1 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %mx = arith.maximumf %in, %out : f32
      linalg.yield %mx : f32
    } -> tensor<64xf32>
    %ins = tensor.insert_slice %m into %arg2[0] [64] [1] : tensor<64xf32> into tensor<64xf32>
    scf.yield %ins : tensor<64xf32>
  } {__reduction_loop__}

  // E: `p = exp(x - m)`, read by BOTH R2 and the divide below.
  %3 = tensor.empty() : tensor<64x512xf32>
  %4 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %2 : tensor<64x512xf32>, tensor<64xf32>) outs(%3 : tensor<64x512xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %d = arith.subf %in, %in_1 : f32
    %e = math.exp %d : f32
    linalg.yield %e : f32
  } -> tensor<64x512xf32>

  // R2: `s = sum_j p`.
  %5 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %6 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%4 : tensor<64x512xf32>) outs(%5 : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %a = arith.addf %in, %out : f32
    linalg.yield %a : f32
  } -> tensor<64xf32>

  // The normalizing divide, downstream of the chain.
  %7 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%4, %6 : tensor<64x512xf32>, tensor<64xf32>) outs(%3 : tensor<64x512xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %d = arith.divf %in, %in_1 : f32
    linalg.yield %d : f32
  } -> tensor<64x512xf32>
  return %7 : tensor<64x512xf32>
}

// The loop is as in @softmax; result #0 is the final max, #1 the clone's
// full-extent result and #2 the running sum.
// CHECK:         %[[LOOP:.+]]:3 = scf.for
// CHECK:           math.exp
// CHECK:           arith.addf
// CHECK:         } {fused_reduction_loop}

// The ORIGINAL E is untouched outside the loop, reading the FINAL max (#0).
// CHECK:         %[[POUT:.+]] = linalg.generic
// CHECK-SAME:        ins(%[[X]], %[[LOOP]]#0 : tensor<64x512xf32>, tensor<64xf32>)
// CHECK:           arith.subf
// CHECK:           math.exp

// The divide reads that, NOT the clone's stale loop result #1.
// CHECK:         linalg.generic
// CHECK-SAME:        ins(%[[POUT]], %[[LOOP]]#2 : tensor<64x512xf32>, tensor<64xf32>)
// CHECK:           arith.divf

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %gen = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // #1 = max (R1, already tiled), #2 = exp term (E), #3 = sum (R2),
    // #4 = normalizing divide.
    %max_op, %e_op, %sum_op, %div_op = transform.split_handle %gen
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op,
                                  !transform.any_op, !transform.any_op)
    %loop = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %fused = transform.structured.fuse_dependant_reduction_ops %e_op, %sum_op into %loop
        : (!transform.any_op, !transform.any_op, !transform.any_op) -> !transform.any_op
    transform.annotate %fused "fused_reduction_loop" : !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#mapik = affine_map<(d0, d1, d2) -> (d0, d2)>
#mapkj = affine_map<(d0, d1, d2) -> (d2, d1)>
#mapij = affine_map<(d0, d1, d2) -> (d0, d1)>

// An attention-shaped chain: one `E` feeding TWO consumer reductions over the same
// axis (a row sum and a `P @ V` contraction), fused into one loop by applying the
// op twice. The first application fuses a clone and leaves `E` for the
// contraction; by the second, `E` has no other consumer, so it is fused itself.
// CHECK-LABEL: func.func @one_elementwise_term_two_consumer_reductions
// CHECK-SAME:      %[[X:[a-zA-Z0-9_]+]]: tensor<64x512xf32>
// CHECK-SAME:      %[[V:[a-zA-Z0-9_]+]]: tensor<512x128xf32>
func.func @one_elementwise_term_two_consumer_reductions(
    %arg0: tensor<64x512xf32>, %v: tensor<512x128xf32>) -> tensor<64x128xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %ninf = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %c32 = arith.constant 32 : index
  %rowInit = tensor.empty() : tensor<64xf32>
  %mInit = linalg.fill ins(%ninf : f32) outs(%rowInit : tensor<64xf32>) -> tensor<64xf32>

  // R1: `max` reduction, already tiled along the reduction axis.
  %m = scf.for %iv = %c0 to %c512 step %c32 iter_args(%acc = %mInit) -> (tensor<64xf32>) {
    %xt = tensor.extract_slice %arg0[0, %iv] [64, 32] [1, 1] : tensor<64x512xf32> to tensor<64x32xf32>
    %mt = tensor.extract_slice %acc[0] [64] [1] : tensor<64xf32> to tensor<64xf32>
    %r = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%xt : tensor<64x32xf32>) outs(%mt : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %mx = arith.maximumf %in, %out : f32
      linalg.yield %mx : f32
    } -> tensor<64xf32>
    %i = tensor.insert_slice %r into %acc[0] [64] [1] : tensor<64xf32> into tensor<64xf32>
    scf.yield %i : tensor<64xf32>
  } {__reduction_loop__}

  // E: a single elementwise term, read by both consumer reductions below.
  %pInit = tensor.empty() : tensor<64x512xf32>
  %p = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %m : tensor<64x512xf32>, tensor<64xf32>) outs(%pInit : tensor<64x512xf32>) {
  ^bb0(%in: f32, %mv: f32, %out: f32):
    %d = arith.subf %in, %mv : f32
    %e = math.exp %d : f32
    linalg.yield %e : f32
  } -> tensor<64x512xf32>

  // R2a: row sum of E.
  %lInit = linalg.fill ins(%zero : f32) outs(%rowInit : tensor<64xf32>) -> tensor<64xf32>
  %l = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%p : tensor<64x512xf32>) outs(%lInit : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %a = arith.addf %in, %out : f32
    linalg.yield %a : f32
  } -> tensor<64xf32>

  // R2b: a contraction over E along the same axis.
  %oInit = tensor.empty() : tensor<64x128xf32>
  %oFill = linalg.fill ins(%zero : f32) outs(%oInit : tensor<64x128xf32>) -> tensor<64x128xf32>
  %o = linalg.generic {indexing_maps = [#mapik, #mapkj, #mapij], iterator_types = ["parallel", "parallel", "reduction"]} ins(%p, %v : tensor<64x512xf32>, tensor<512x128xf32>) outs(%oFill : tensor<64x128xf32>) {
  ^bb0(%pv: f32, %vv: f32, %out: f32):
    %mul = arith.mulf %pv, %vv : f32
    %add = arith.addf %out, %mul : f32
    linalg.yield %add : f32
  } -> tensor<64x128xf32>

  // The deferred normalization, downstream of both reductions.
  %outInit = tensor.empty() : tensor<64x128xf32>
  %out = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%o, %l : tensor<64x128xf32>, tensor<64xf32>) outs(%outInit : tensor<64x128xf32>) {
  ^bb0(%in: f32, %lv: f32, %o2: f32):
    %d = arith.divf %in, %lv : f32
    linalg.yield %d : f32
  } -> tensor<64x128xf32>
  return %out : tensor<64x128xf32>
}

// The loop carries five accumulators: the running max, one full-extent E result
// per fused chain, the running sum and the running contraction accumulator.
// CHECK:         %[[LOOP:.+]]:5 = scf.for %[[IV:[a-zA-Z0-9_]+]] =
// CHECK-SAME:        iter_args(%[[MARG:[a-zA-Z0-9_]+]] = %{{[a-zA-Z0-9_]+}}, %[[E1ARG:[a-zA-Z0-9_]+]] = %{{[a-zA-Z0-9_]+}}, %[[LARG:[a-zA-Z0-9_]+]] = %{{[a-zA-Z0-9_]+}}, %[[E2ARG:[a-zA-Z0-9_]+]] = %{{[a-zA-Z0-9_]+}}, %[[OARG:[a-zA-Z0-9_]+]] = %{{[a-zA-Z0-9_]+}})
// CHECK-SAME:        -> (tensor<64xf32>, tensor<64x512xf32>, tensor<64xf32>, tensor<64x512xf32>, tensor<64x128xf32>)

// CHECK:           %[[MOLD:.+]] = tensor.extract_slice %[[MARG]][0] [64] [1]
// CHECK:           %[[M:.+]] = linalg.generic
// CHECK-SAME:          outs(%[[MOLD]] :
// CHECK:             arith.maximumf

// The clone, fused for the row sum, and the original E, fused for the
// contraction. Both read the same new max and are cut to the [64, 32] tile.
// CHECK:           %[[E1DST:.+]] = tensor.extract_slice %[[E1ARG]][0, %[[IV]]] [64, 32] [1, 1]
// CHECK:           %[[E1:.+]] = linalg.generic
// CHECK-SAME:          ins(%{{.+}}, %[[M]] : tensor<64x32xf32>, tensor<64xf32>)
// CHECK-SAME:          outs(%[[E1DST]] : tensor<64x32xf32>)
// CHECK:             math.exp
// CHECK:           %[[E2DST:.+]] = tensor.extract_slice %[[E2ARG]][0, %[[IV]]] [64, 32] [1, 1]
// CHECK:           %[[E2:.+]] = linalg.generic
// CHECK-SAME:          ins(%{{.+}}, %[[M]] : tensor<64x32xf32>, tensor<64xf32>)
// CHECK-SAME:          outs(%[[E2DST]] : tensor<64x32xf32>)
// CHECK:             math.exp

// Each fused consumer reduction gets its own online correction, built over its
// own accumulator shape: [64] for the row sum, [64x128] for the contraction.
// CHECK:           %[[F1:.+]] = linalg.elementwise kind=#linalg.elementwise_kind<div> ins(%{{.+}}, %{{.+}} : tensor<64xf32>, tensor<64xf32>)
// CHECK:           %[[L_SCALED:.+]] = linalg.elementwise kind=#linalg.elementwise_kind<mul> ins(%{{.+}}, %[[F1]] : tensor<64xf32>, tensor<64xf32>)
// CHECK:           %[[L:.+]] = linalg.generic
// CHECK-SAME:          ins(%[[E1]] : tensor<64x32xf32>)
// CHECK-SAME:          outs(%[[L_SCALED]] : tensor<64xf32>)
// CHECK:             arith.addf
// CHECK:           %[[VTILE:.+]] = tensor.extract_slice %[[V]][%[[IV]], 0] [32, 128] [1, 1]
// CHECK:           %[[F2:.+]] = linalg.elementwise kind=#linalg.elementwise_kind<div> ins(%{{.+}}, %{{.+}} : tensor<64x128xf32>, tensor<64x128xf32>)
// CHECK:           %[[O_SCALED:.+]] = linalg.elementwise kind=#linalg.elementwise_kind<mul> ins(%{{.+}}, %[[F2]] : tensor<64x128xf32>, tensor<64x128xf32>)
// CHECK:           %[[O:.+]] = linalg.generic
// CHECK-SAME:          ins(%[[E2]], %[[VTILE]] : tensor<64x32xf32>, tensor<32x128xf32>)
// CHECK-SAME:          outs(%[[O_SCALED]] : tensor<64x128xf32>)
// CHECK:             arith.mulf
// CHECK:             arith.addf
// CHECK:         } {fused_reduction_loop}

// Only the normalization is left outside, reading the two running results.
// CHECK:         linalg.generic
// CHECK-SAME:        ins(%[[LOOP]]#4, %[[LOOP]]#2 : tensor<64x128xf32>, tensor<64xf32>)
// CHECK:           arith.divf

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %gen = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // #1 = max (R1), #2 = E, #3 = row sum (R2a), #4 = contraction (R2b),
    // #5 = the normalizing divide.
    %r1, %e, %r2a, %r2b, %div = transform.split_handle %gen
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op,
                                  !transform.any_op, !transform.any_op)
    %loop = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %fused_a = transform.structured.fuse_dependant_reduction_ops %e, %r2a into %loop
        : (!transform.any_op, !transform.any_op, !transform.any_op) -> !transform.any_op
    // The fused loop is the producer reduction of the second chain.
    transform.annotate %fused_a "__reduction_loop__" : !transform.any_op
    // The first fusion consumed the handle to E; the original E is still the
    // contraction's operand, so re-derive it from there.
    %e_again = transform.get_producer_of_operand %r2b[0] : (!transform.any_op) -> !transform.any_op
    %fused_b = transform.structured.fuse_dependant_reduction_ops %e_again, %r2b into %fused_a
        : (!transform.any_op, !transform.any_op, !transform.any_op) -> !transform.any_op
    transform.annotate %fused_b "fused_reduction_loop" : !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

// Two independent softmax chains. The handles name chain A only, so the control
// function must confine the fusion to it and leave chain B alone -- the pass
// would fuse both.
// CHECK-LABEL: func.func @only_the_handled_chain_is_fused
func.func @only_the_handled_chain_is_fused(%arg0: tensor<64x512xf32>, %arg1: tensor<64x512xf32>)
    -> (tensor<64xf32>, tensor<64xf32>) {
  %zero = arith.constant 0.000000e+00 : f32
  %ninf = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %c32 = arith.constant 32 : index
  %empty = tensor.empty() : tensor<64xf32>
  %negInit = linalg.fill ins(%ninf : f32) outs(%empty : tensor<64xf32>) -> tensor<64xf32>

  // Chain A.
  %ma = scf.for %iv = %c0 to %c512 step %c32 iter_args(%acc = %negInit) -> (tensor<64xf32>) {
    %s = tensor.extract_slice %arg0[0, %iv] [64, 32] [1, 1] : tensor<64x512xf32> to tensor<64x32xf32>
    %o = tensor.extract_slice %acc[0] [64] [1] : tensor<64xf32> to tensor<64xf32>
    %r = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%s : tensor<64x32xf32>) outs(%o : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %v = arith.maximumf %in, %out : f32
      linalg.yield %v : f32
    } -> tensor<64xf32>
    %i = tensor.insert_slice %r into %acc[0] [64] [1] : tensor<64xf32> into tensor<64xf32>
    scf.yield %i : tensor<64xf32>
  } {__reduction_loop__}
  %ea = tensor.empty() : tensor<64x512xf32>
  %pa = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %ma : tensor<64x512xf32>, tensor<64xf32>) outs(%ea : tensor<64x512xf32>) {
  ^bb0(%in: f32, %m: f32, %out: f32):
    %d = arith.subf %in, %m : f32
    %e = math.exp %d : f32
    linalg.yield %e : f32
  } -> tensor<64x512xf32>
  %za = linalg.fill ins(%zero : f32) outs(%empty : tensor<64xf32>) -> tensor<64xf32>
  %sa = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%pa : tensor<64x512xf32>) outs(%za : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %v = arith.addf %in, %out : f32
    linalg.yield %v : f32
  } -> tensor<64xf32>

  // Chain B: structurally identical, over a different input.
  %mb = scf.for %iv = %c0 to %c512 step %c32 iter_args(%acc = %negInit) -> (tensor<64xf32>) {
    %s = tensor.extract_slice %arg1[0, %iv] [64, 32] [1, 1] : tensor<64x512xf32> to tensor<64x32xf32>
    %o = tensor.extract_slice %acc[0] [64] [1] : tensor<64xf32> to tensor<64xf32>
    %r = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%s : tensor<64x32xf32>) outs(%o : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %v = arith.maximumf %in, %out : f32
      linalg.yield %v : f32
    } -> tensor<64xf32>
    %i = tensor.insert_slice %r into %acc[0] [64] [1] : tensor<64xf32> into tensor<64xf32>
    scf.yield %i : tensor<64xf32>
  } {__reduction_loop__}
  %eb = tensor.empty() : tensor<64x512xf32>
  %pb = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%arg1, %mb : tensor<64x512xf32>, tensor<64xf32>) outs(%eb : tensor<64x512xf32>) {
  ^bb0(%in: f32, %m: f32, %out: f32):
    %d = arith.subf %in, %m : f32
    %e = math.exp %d : f32
    linalg.yield %e : f32
  } -> tensor<64x512xf32>
  %zb = linalg.fill ins(%zero : f32) outs(%empty : tensor<64xf32>) -> tensor<64xf32>
  %sb = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%pb : tensor<64x512xf32>) outs(%zb : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %v = arith.addf %in, %out : f32
    linalg.yield %v : f32
  } -> tensor<64xf32>

  return %sa, %sb : tensor<64xf32>, tensor<64xf32>
}

// Chain A's loop is fused: three accumulators, correction inside.
// CHECK:         %[[A:.+]]:3 = scf.for
// CHECK-SAME:        -> (tensor<64xf32>, tensor<64x512xf32>, tensor<64xf32>)
// CHECK:           arith.maximumf
// CHECK:           math.exp
// CHECK:           linalg.elementwise kind=#linalg.elementwise_kind<div>
// CHECK:           linalg.elementwise kind=#linalg.elementwise_kind<mul>
// CHECK:           arith.addf
// CHECK:         } {fused_reduction_loop}

// Chain B's loop is untouched: one accumulator, holding only its R1, and still
// annotated as an un-fused reduction loop.
// CHECK:         %[[B:.+]] = scf.for {{.*}} -> (tensor<64xf32>) {
// CHECK:           linalg.generic
// CHECK:             arith.maximumf
// CHECK-NOT:       linalg.generic
// CHECK:         } {__reduction_loop__}
// Chain B's E and R2 remain outside the loop.
// CHECK:         linalg.generic
// CHECK:           math.exp
// CHECK:         linalg.generic
// CHECK:           arith.addf
// CHECK:         return %[[A]]#2, %{{.+}}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %gen = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %r1_a, %e_a, %r2_a, %r1_b, %e_b, %r2_b = transform.split_handle %gen
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op,
                                  !transform.any_op, !transform.any_op, !transform.any_op)
    %loops = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %loop_a, %loop_b = transform.split_handle %loops
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused = transform.structured.fuse_dependant_reduction_ops %e_a, %r2_a into %loop_a
        : (!transform.any_op, !transform.any_op, !transform.any_op) -> !transform.any_op
    transform.annotate %fused "fused_reduction_loop" : !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

// An illegal triple -- `R2` combines with `maximumf` rather than `addf`, so the
// online correction does not apply -- is reported as a silenceable failure
// rather than silently leaving the payload alone.
func.func @illegal_triple_is_reported(%arg0: tensor<64x512xf32>) -> tensor<64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %c32 = arith.constant 32 : index
  %0 = tensor.empty() : tensor<64xf32>
  %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %2 = scf.for %arg1 = %c0 to %c512 step %c32 iter_args(%arg2 = %1) -> (tensor<64xf32>) {
    %es = tensor.extract_slice %arg0[0, %arg1] [64, 32] [1, 1] : tensor<64x512xf32> to tensor<64x32xf32>
    %es1 = tensor.extract_slice %arg2[0] [64] [1] : tensor<64xf32> to tensor<64xf32>
    %m = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%es : tensor<64x32xf32>) outs(%es1 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %mx = arith.maximumf %in, %out : f32
      linalg.yield %mx : f32
    } -> tensor<64xf32>
    %ins = tensor.insert_slice %m into %arg2[0] [64] [1] : tensor<64xf32> into tensor<64xf32>
    scf.yield %ins : tensor<64xf32>
  } {__reduction_loop__}
  %3 = tensor.empty() : tensor<64x512xf32>
  %4 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %2 : tensor<64x512xf32>, tensor<64xf32>) outs(%3 : tensor<64x512xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %d = arith.subf %in, %in_1 : f32
    %e = math.exp %d : f32
    linalg.yield %e : f32
  } -> tensor<64x512xf32>
  %5 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %6 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%4 : tensor<64x512xf32>) outs(%5 : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %a = arith.maximumf %in, %out : f32
    linalg.yield %a : f32
  } -> tensor<64xf32>
  return %6 : tensor<64xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %gen = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %r1, %e_op, %r2 = transform.split_handle %gen
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %loop = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{could not fuse the elementwise op and the consumer reduction into the producer reduction loop}}
    %fused = transform.structured.fuse_dependant_reduction_ops %e_op, %r2 into %loop
        : (!transform.any_op, !transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#mapik = affine_map<(d0, d1, d2) -> (d0, d2)>
#mapkj = affine_map<(d0, d1, d2) -> (d2, d1)>
#mapij = affine_map<(d0, d1, d2) -> (d0, d1)>

// An `R2` operand defined *below* the reduction loop: tiling an enclosing
// parallel dim leaves the `tensor.extract_slice` feeding a data input just
// before its consumer, hence after the loop. `tileAndFuseConsumer` moves the
// loop to just before the op being fused, so such an operand has to be hoisted
// above the loop first or only `E` gets fused.
// CHECK-LABEL: func.func @r2_input_defined_below_the_loop
// CHECK-SAME:      %[[X:[a-zA-Z0-9_]+]]: tensor<64x512xf32>
// CHECK-SAME:      %[[V:[a-zA-Z0-9_]+]]: tensor<2x512x128xf32>
func.func @r2_input_defined_below_the_loop(%arg0: tensor<64x512xf32>, %v: tensor<2x512x128xf32>)
    -> tensor<64x128xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %ninf = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %c32 = arith.constant 32 : index
  %rowInit = tensor.empty() : tensor<64xf32>
  %mInit = linalg.fill ins(%ninf : f32) outs(%rowInit : tensor<64xf32>) -> tensor<64xf32>
  %m = scf.for %iv = %c0 to %c512 step %c32 iter_args(%acc = %mInit) -> (tensor<64xf32>) {
    %xt = tensor.extract_slice %arg0[0, %iv] [64, 32] [1, 1] : tensor<64x512xf32> to tensor<64x32xf32>
    %mt = tensor.extract_slice %acc[0] [64] [1] : tensor<64xf32> to tensor<64xf32>
    %r = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%xt : tensor<64x32xf32>) outs(%mt : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %mx = arith.maximumf %in, %out : f32
      linalg.yield %mx : f32
    } -> tensor<64xf32>
    %i = tensor.insert_slice %r into %acc[0] [64] [1] : tensor<64xf32> into tensor<64xf32>
    scf.yield %i : tensor<64xf32>
  } {__reduction_loop__}
  %pInit = tensor.empty() : tensor<64x512xf32>
  %p = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %m : tensor<64x512xf32>, tensor<64xf32>) outs(%pInit : tensor<64x512xf32>) {
  ^bb0(%in: f32, %mv: f32, %out: f32):
    %d = arith.subf %in, %mv : f32
    %e = math.exp %d : f32
    linalg.yield %e : f32
  } -> tensor<64x512xf32>

  // R2's `v` input is sliced *below* the loop, which is where tiling an
  // enclosing parallel dim leaves it. The fusion has to hoist this definition
  // above the loop, otherwise `tileAndFuseConsumer` cannot move the loop to
  // just before R2 and only E would end up fused.
  %vs = tensor.extract_slice %v[1, 0, 0] [1, 512, 128] [1, 1, 1] : tensor<2x512x128xf32> to tensor<512x128xf32>
  %oInit = tensor.empty() : tensor<64x128xf32>
  %oFill = linalg.fill ins(%zero : f32) outs(%oInit : tensor<64x128xf32>) -> tensor<64x128xf32>
  %o = linalg.generic {indexing_maps = [#mapik, #mapkj, #mapij], iterator_types = ["parallel", "parallel", "reduction"]} ins(%p, %vs : tensor<64x512xf32>, tensor<512x128xf32>) outs(%oFill : tensor<64x128xf32>) {
  ^bb0(%pv: f32, %vv: f32, %out: f32):
    %mul = arith.mulf %pv, %vv : f32
    %add = arith.addf %out, %mul : f32
    linalg.yield %add : f32
  } -> tensor<64x128xf32>
  return %o : tensor<64x128xf32>
}

// The `v` slice is hoisted above the loop, which now carries all three
// accumulators -- proving the contraction was fused and not left behind.
// CHECK:         %[[VS:.+]] = tensor.extract_slice %[[V]][1, 0, 0] [1, 512, 128] [1, 1, 1]
// CHECK:         %[[LOOP:.+]]:3 = scf.for %[[IV:[a-zA-Z0-9_]+]] = %{{[a-zA-Z0-9_]+}} to %{{[a-zA-Z0-9_]+}} step %{{[a-zA-Z0-9_]+}}
// CHECK-SAME:        -> (tensor<64xf32>, tensor<64x512xf32>, tensor<64x128xf32>)
// CHECK:           arith.maximumf
// CHECK:           math.exp
// CHECK:           tensor.extract_slice %[[VS]][%[[IV]], 0] [32, 128] [1, 1]
// CHECK:           linalg.elementwise kind=#linalg.elementwise_kind<div>
// CHECK:           linalg.elementwise kind=#linalg.elementwise_kind<mul>
// CHECK:           linalg.generic
// CHECK:             arith.mulf
// CHECK:             arith.addf
// CHECK:         } {fused_reduction_loop}
// CHECK:         return %[[LOOP]]#2

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %gen = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %r1, %e, %r2 = transform.split_handle %gen
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %loop = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %fused = transform.structured.fuse_dependant_reduction_ops %e, %r2 into %loop
        : (!transform.any_op, !transform.any_op, !transform.any_op) -> !transform.any_op
    transform.annotate %fused "fused_reduction_loop" : !transform.any_op
    transform.yield
  }
}
