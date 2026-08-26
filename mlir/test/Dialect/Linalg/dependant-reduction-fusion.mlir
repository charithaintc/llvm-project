// RUN: mlir-opt --test-linalg-dependant-reduction-fusion -split-input-file %s | FileCheck %s

// Positive cases fuse E and R2 into the R1 loop; negative ones leave the chain
// outside it. Run with `--debug-only=dependant-reduction-fusion` to see which
// legality condition rejected a negative case.

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: func.func @softmax
// CHECK-DAG:     %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:     %[[NEGINF:.+]] = arith.constant 0xFF800000 : f32
// CHECK:         %[[X:.+]] = bufferization.to_tensor %{{.+}} restrict :
// CHECK:         %[[DST:.+]] = bufferization.to_tensor %{{.+}} restrict writable :
// CHECK:         %[[MINIT:.+]] = linalg.fill ins(%[[NEGINF]]
// R2's init fill is hoisted above the loop.
// CHECK:         %[[SINIT:.+]] = linalg.fill ins(%[[ZERO]]
// The loop carries three accumulators: `m`, the fused E clone's full result and
// `s`.
// CHECK:         %[[LOOP:.+]]:3 = scf.for %[[IV:[a-zA-Z0-9_]+]] =
// CHECK-SAME:        iter_args(%[[MARG:.+]] = %[[MINIT]], %[[EARG:.+]] = %[[DST]], %[[SARG:.+]] = %[[SINIT]])
// CHECK-SAME:        -> (tensor<64xf32>, tensor<64x512xf32>, tensor<64xf32>)
// R1 reads the current tile of `x`.
// CHECK:           %[[XTILE:.+]] = tensor.extract_slice %[[X]][0, %[[IV]]] [64, 32] [1, 1]
// R1's DPS init holds the previous running `m`; the "old" term reads it.
// CHECK:           %[[MOLD:.+]] = tensor.extract_slice %[[MARG]][0] [64] [1]
// CHECK:           %[[M:.+]] = linalg.generic
// CHECK-SAME:          ins(%[[XTILE]] : tensor<64x32xf32>)
// CHECK-SAME:          outs(%[[MOLD]] : tensor<64xf32>)
// CHECK:             arith.maximumf
// E's clone is fused and re-sliced to the tile: both its `x` input and its
// destination are cut to [64, 32], so its result is a [64, 32] tile.
// CHECK:           %[[XTILE2:.+]] = tensor.extract_slice %[[X]][0, %[[IV]]] [64, 32] [1, 1]
// CHECK:           %[[EDST:.+]] = tensor.extract_slice %[[EARG]][0, %[[IV]]] [64, 32] [1, 1]
// CHECK:           %[[E:.+]] = linalg.generic
// CHECK-SAME:          ins(%[[XTILE2]], %[[M]] : tensor<64x32xf32>, tensor<64xf32>)
// CHECK-SAME:          outs(%[[EDST]] : tensor<64x32xf32>)
// CHECK:             arith.subf
// CHECK:             math.exp
// CHECK:           } -> tensor<64x32xf32>
// R2 is fused, reducing E's [64, 32] tile into the carried `s`. Its output does
// not span the tiled axis, so `s` keeps its full [64] shape.
// CHECK:           %[[SSLICE:.+]] = tensor.extract_slice %[[SARG]][0] [64] [1]
// The factor is term(new)/term(old) -- E isolated on `m`, evaluated with the
// updated and the previous `m` -- and rescales the running `s` before this
// tile is accumulated into it.
// CHECK:           %[[TERMNEW:.+]] = linalg.generic
// CHECK-SAME:          ins(%[[M]] : tensor<64xf32>)
// CHECK:             math.exp
// CHECK:           %[[TERMOLD:.+]] = linalg.generic
// CHECK-SAME:          ins(%[[MOLD]] : tensor<64xf32>)
// CHECK:             math.exp
// CHECK:           %[[FACTOR:.+]] = linalg.elementwise kind=#linalg.elementwise_kind<div> ins(%[[TERMNEW]], %[[TERMOLD]] : tensor<64xf32>, tensor<64xf32>)
// CHECK:           %[[SCALED:.+]] = linalg.elementwise kind=#linalg.elementwise_kind<mul> ins(%[[SSLICE]], %[[FACTOR]] : tensor<64xf32>, tensor<64xf32>)
// CHECK:           %[[S:.+]] = linalg.generic
// CHECK-SAME:          ins(%[[E]] : tensor<64x32xf32>)
// CHECK-SAME:          outs(%[[SCALED]] : tensor<64xf32>)
// CHECK:             arith.addf
// The clone's tile is inserted back at the loop IV. Nothing reads that loop
// result -- it is dead, and `--remove-dead-values` unwinds it.
// CHECK:           tensor.insert_slice %[[E]] into %[[EARG]][0, %[[IV]]] [64, 32] [1, 1]
// CHECK:           tensor.insert_slice %[[S]] into %[[SARG]][0] [64] [1]
// CHECK:           scf.yield
// The ORIGINAL E stays outside, recomputing `p = exp(x - m)` from the *final*
// `m`. Sharing the fused copy instead would hand the divide numerators computed
// against the running `m`, which the correction only rescales for `s`.
// CHECK:         %[[POUT:.+]] = linalg.generic
// CHECK-SAME:        ins(%[[X]], %[[LOOP]]#0 : tensor<64x512xf32>, tensor<64xf32>)
// CHECK:           arith.subf
// CHECK:           math.exp
// The divide reads that full-extent `p` and R2's loop result.
// CHECK:         linalg.generic
// CHECK-SAME:        ins(%[[POUT]], %[[LOOP]]#2 : tensor<64x512xf32>, tensor<64xf32>)
// CHECK:           arith.divf
func.func @softmax(%arg0: memref<64x512xf32>, %arg1: memref<64x512xf32>) {
    %0 = bufferization.to_tensor %arg0 restrict : memref<64x512xf32> to tensor<64x512xf32>
    %1 = bufferization.to_tensor %arg1 restrict writable : memref<64x512xf32> to tensor<64x512xf32>
    %2 = tensor.empty() : tensor<64xf32>
    %cst = arith.constant 0xFF800000 : f32
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<64xf32>) -> tensor<64xf32>
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %c32 = arith.constant 32 : index
    // R1: tiled `max` reduction carrying the running `m` as an iter_arg.
    %4 = scf.for %arg2 = %c0 to %c512 step %c32 iter_args(%arg3 = %3) -> (tensor<64xf32>) {
      %extracted_slice = tensor.extract_slice %0[0, %arg2] [64, 32] [1, 1] : tensor<64x512xf32> to tensor<64x32xf32>
      %extracted_slice_1 = tensor.extract_slice %arg3[0] [64] [1] : tensor<64xf32> to tensor<64xf32>
      %9 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice : tensor<64x32xf32>) outs(%extracted_slice_1 : tensor<64xf32>) {
      ^bb0(%in: f32, %out: f32):
        %10 = arith.maximumf %in, %out : f32
        linalg.yield %10 : f32
      } -> tensor<64xf32>
      %inserted_slice = tensor.insert_slice %9 into %arg3[0] [64] [1] : tensor<64xf32> into tensor<64xf32>
      scf.yield %inserted_slice : tensor<64xf32>
    } {__reduction_loop__}
    // E: all-parallel elementwise term `p = exp(x - m)`.
    %5 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%0, %4 : tensor<64x512xf32>, tensor<64xf32>) outs(%1 : tensor<64x512xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %9 = arith.subf %in, %in_1 : f32
      %10 = math.exp %9 : f32
      linalg.yield %10 : f32
    } -> tensor<64x512xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %6 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<64xf32>) -> tensor<64xf32>
    // R2: pure single-input add-reduction `s = sum_j p`.
    %7 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%5 : tensor<64x512xf32>) outs(%6 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = arith.addf %in, %out : f32
      linalg.yield %9 : f32
    } -> tensor<64xf32>
    // Normalizing divide -- downstream of the chain, not part of it.
    %8 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%5, %7 : tensor<64x512xf32>, tensor<64xf32>) outs(%1 : tensor<64x512xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %9 = arith.divf %in, %in_1 : f32
      linalg.yield %9 : f32
    } -> tensor<64x512xf32>
    bufferization.materialize_in_destination %8 in restrict writable %arg1 : (tensor<64x512xf32>, memref<64x512xf32>) -> ()
    return
  }

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: func.func @r2_two_inputs_both_reduced
// CHECK-SAME:      (%[[X:.+]]: tensor<64x512xf32>, %[[DST:.+]]: tensor<64x512xf32>)
// CHECK-DAG:     %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:     %[[NEGINF:.+]] = arith.constant 0xFF800000 : f32
// CHECK:         %[[MINIT:.+]] = linalg.fill ins(%[[NEGINF]]
// CHECK:         %[[SINIT:.+]] = linalg.fill ins(%[[ZERO]]
// CHECK:         %[[LOOP:.+]]:3 = scf.for %[[IV:[a-zA-Z0-9_]+]] =
// CHECK-SAME:        iter_args(%[[MARG:.+]] = %[[MINIT]], %[[EARG:.+]] = %[[DST]], %[[SARG:.+]] = %[[SINIT]])
// CHECK-SAME:        -> (tensor<64xf32>, tensor<64x512xf32>, tensor<64xf32>)
// CHECK:           %[[XTILE:.+]] = tensor.extract_slice %[[X]][0, %[[IV]]] [64, 32] [1, 1]
// CHECK:           %[[MOLD:.+]] = tensor.extract_slice %[[MARG]][0] [64] [1]
// CHECK:           %[[M:.+]] = linalg.generic
// CHECK-SAME:          ins(%[[XTILE]] : tensor<64x32xf32>)
// CHECK-SAME:          outs(%[[MOLD]] : tensor<64xf32>)
// CHECK:             arith.maximumf
// CHECK:           %[[E:.+]] = linalg.generic
// CHECK-SAME:          tensor<64x32xf32>, tensor<64xf32>
// CHECK:             arith.subf
// CHECK:             math.exp
// Both R2 inputs -- E's tile and `x` -- are re-sliced to the tile.
// CHECK:           %[[XTILE2:.+]] = tensor.extract_slice %[[X]][0, %[[IV]]] [64, 32] [1, 1]
// CHECK:           %[[SSLICE:.+]] = tensor.extract_slice %[[SARG]][0] [64] [1]
// CHECK:           %[[TERMNEW:.+]] = linalg.generic
// CHECK-SAME:          ins(%[[M]] : tensor<64xf32>)
// CHECK:             math.exp
// CHECK:           %[[TERMOLD:.+]] = linalg.generic
// CHECK-SAME:          ins(%[[MOLD]] : tensor<64xf32>)
// CHECK:             math.exp
// CHECK:           %[[FACTOR:.+]] = linalg.elementwise kind=#linalg.elementwise_kind<div> ins(%[[TERMNEW]], %[[TERMOLD]] : tensor<64xf32>, tensor<64xf32>)
// CHECK:           %[[SCALED:.+]] = linalg.elementwise kind=#linalg.elementwise_kind<mul> ins(%[[SSLICE]], %[[FACTOR]] : tensor<64xf32>, tensor<64xf32>)
// CHECK:           %[[R2:.+]] = linalg.generic
// CHECK-SAME:          ins(%[[E]], %[[XTILE2]] : tensor<64x32xf32>, tensor<64x32xf32>)
// CHECK-SAME:          outs(%[[SCALED]] : tensor<64xf32>)
// CHECK:             arith.mulf
// CHECK:             arith.addf
// CHECK:           scf.yield
// CHECK:         return %[[LOOP]]#2
func.func @r2_two_inputs_both_reduced(%arg0: tensor<64x512xf32>, %argd: tensor<64x512xf32>) -> tensor<64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %cstn = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %c32 = arith.constant 32 : index
  %0 = tensor.empty() : tensor<64xf32>
  %1 = linalg.fill ins(%cstn : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %2 = scf.for %a = %c0 to %c512 step %c32 iter_args(%b = %1) -> (tensor<64xf32>) {
    %es = tensor.extract_slice %arg0[0, %a] [64, 32] [1, 1] : tensor<64x512xf32> to tensor<64x32xf32>
    %es1 = tensor.extract_slice %b[0] [64] [1] : tensor<64xf32> to tensor<64xf32>
    %m = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%es : tensor<64x32xf32>) outs(%es1 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %v = arith.maximumf %in, %out : f32
      linalg.yield %v : f32
    } -> tensor<64xf32>
    %ins = tensor.insert_slice %m into %b[0] [64] [1] : tensor<64xf32> into tensor<64xf32>
    scf.yield %ins : tensor<64xf32>
  } {__reduction_loop__}
  %3 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %2 : tensor<64x512xf32>, tensor<64xf32>) outs(%argd : tensor<64x512xf32>) {
  ^bb0(%in: f32, %mv: f32, %out: f32):
    %d = arith.subf %in, %mv : f32
    %v = math.exp %d : f32
    linalg.yield %v : f32
  } -> tensor<64x512xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %5 = linalg.generic {indexing_maps = [#map, #map, #map1], iterator_types = ["parallel", "reduction"]} ins(%3, %arg0 : tensor<64x512xf32>, tensor<64x512xf32>) outs(%4 : tensor<64xf32>) {
  ^bb0(%in: f32, %in2: f32, %out: f32):
    %p = arith.mulf %in, %in2 : f32
    %v = arith.addf %p, %out : f32
    linalg.yield %v : f32
  } -> tensor<64xf32>
  return %5 : tensor<64xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: func.func @neg_e_not_sharing_r1_input
// The loop is unchanged: it carries a single accumulator and holds only R1.
// CHECK:         scf.for %{{.+}} = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%{{.+}} = %{{.+}}) -> (tensor<64xf32>) {
// CHECK:           linalg.generic
// CHECK:             arith.maximumf
// CHECK-NOT:       linalg.generic
// CHECK:         }
// E and R2 remain outside the loop.
// CHECK:         linalg.generic
func.func @neg_e_not_sharing_r1_input(%arg0: tensor<64x512xf32>, %argd: tensor<64x512xf32>) -> tensor<64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %cstn = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %c32 = arith.constant 32 : index
  %0 = tensor.empty() : tensor<64xf32>
  %1 = linalg.fill ins(%cstn : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %2 = scf.for %a = %c0 to %c512 step %c32 iter_args(%b = %1) -> (tensor<64xf32>) {
    %es = tensor.extract_slice %arg0[0, %a] [64, 32] [1, 1] : tensor<64x512xf32> to tensor<64x32xf32>
    %es1 = tensor.extract_slice %b[0] [64] [1] : tensor<64xf32> to tensor<64xf32>
    %m = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%es : tensor<64x32xf32>) outs(%es1 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %v = arith.maximumf %in, %out : f32
      linalg.yield %v : f32
    } -> tensor<64xf32>
    %ins = tensor.insert_slice %m into %b[0] [64] [1] : tensor<64xf32> into tensor<64xf32>
    scf.yield %ins : tensor<64xf32>
  } {__reduction_loop__}
  %3 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%argd, %2 : tensor<64x512xf32>, tensor<64xf32>) outs(%argd : tensor<64x512xf32>) {
  ^bb0(%in: f32, %mv: f32, %out: f32):
    %d = arith.subf %in, %mv : f32
    %v = math.exp %d : f32
    linalg.yield %v : f32
  } -> tensor<64x512xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %5 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%3 : tensor<64x512xf32>) outs(%4 : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %v = arith.addf %in, %out : f32
    linalg.yield %v : f32
  } -> tensor<64xf32>
  return %5 : tensor<64xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#mapj = affine_map<(d0, d1) -> (d1)>

// CHECK-LABEL: func.func @neg_accumulator_not_broadcast
// The loop is unchanged: it carries a single accumulator and holds only R1.
// CHECK:         scf.for %{{.+}} = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%{{.+}} = %{{.+}}) -> (tensor<512xf32>) {
// CHECK:           linalg.generic
// CHECK:             arith.maximumf
// CHECK-NOT:       linalg.generic
// CHECK:         }
// E and R2 remain outside the loop.
// CHECK:         linalg.generic
func.func @neg_accumulator_not_broadcast(%arg0: tensor<512x512xf32>) -> tensor<512xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %cstn = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %c32 = arith.constant 32 : index
  %0 = tensor.empty() : tensor<512xf32>
  %1 = linalg.fill ins(%cstn : f32) outs(%0 : tensor<512xf32>) -> tensor<512xf32>
  %2 = scf.for %a = %c0 to %c512 step %c32 iter_args(%b = %1) -> (tensor<512xf32>) {
    %es = tensor.extract_slice %arg0[0, %a] [512, 32] [1, 1] : tensor<512x512xf32> to tensor<512x32xf32>
    %es1 = tensor.extract_slice %b[0] [512] [1] : tensor<512xf32> to tensor<512xf32>
    %m = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%es : tensor<512x32xf32>) outs(%es1 : tensor<512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %v = arith.maximumf %in, %out : f32
      linalg.yield %v : f32
    } -> tensor<512xf32>
    %ins = tensor.insert_slice %m into %b[0] [512] [1] : tensor<512xf32> into tensor<512xf32>
    scf.yield %ins : tensor<512xf32>
  } {__reduction_loop__}
  %e0 = tensor.empty() : tensor<512x512xf32>
  %3 = linalg.generic {indexing_maps = [#map, #mapj, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %2 : tensor<512x512xf32>, tensor<512xf32>) outs(%e0 : tensor<512x512xf32>) {
  ^bb0(%in: f32, %mv: f32, %out: f32):
    %d = arith.subf %in, %mv : f32
    %v = math.exp %d : f32
    linalg.yield %v : f32
  } -> tensor<512x512xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%0 : tensor<512xf32>) -> tensor<512xf32>
  %5 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%3 : tensor<512x512xf32>) outs(%4 : tensor<512xf32>) {
  ^bb0(%in: f32, %out: f32):
    %v = arith.addf %in, %out : f32
    linalg.yield %v : f32
  } -> tensor<512xf32>
  return %5 : tensor<512xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: func.func @neg_extent_mismatch
// The loop is unchanged: it carries a single accumulator and holds only R1.
// CHECK:         scf.for %{{.+}} = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%{{.+}} = %{{.+}}) -> (tensor<64xf32>) {
// CHECK:           linalg.generic
// CHECK:             arith.maximumf
// CHECK-NOT:       linalg.generic
// CHECK:         }
// E and R2 remain outside the loop.
// CHECK:         linalg.generic
func.func @neg_extent_mismatch(%arg0: tensor<64x64xf32>, %argd: tensor<64x512xf32>) -> tensor<64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %cstn = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c32 = arith.constant 32 : index
  %0 = tensor.empty() : tensor<64xf32>
  %1 = linalg.fill ins(%cstn : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %2 = scf.for %a = %c0 to %c64 step %c32 iter_args(%b = %1) -> (tensor<64xf32>) {
    %es = tensor.extract_slice %arg0[0, %a] [64, 32] [1, 1] : tensor<64x64xf32> to tensor<64x32xf32>
    %es1 = tensor.extract_slice %b[0] [64] [1] : tensor<64xf32> to tensor<64xf32>
    %m = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%es : tensor<64x32xf32>) outs(%es1 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %v = arith.maximumf %in, %out : f32
      linalg.yield %v : f32
    } -> tensor<64xf32>
    %ins = tensor.insert_slice %m into %b[0] [64] [1] : tensor<64xf32> into tensor<64xf32>
    scf.yield %ins : tensor<64xf32>
  } {__reduction_loop__}
  %e0 = tensor.empty() : tensor<64x512xf32>
  %3 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%argd, %2 : tensor<64x512xf32>, tensor<64xf32>) outs(%e0 : tensor<64x512xf32>) {
  ^bb0(%in: f32, %mv: f32, %out: f32):
    %d = arith.subf %in, %mv : f32
    %v = math.exp %d : f32
    linalg.yield %v : f32
  } -> tensor<64x512xf32>
  %o = tensor.empty() : tensor<64xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%o : tensor<64xf32>) -> tensor<64xf32>
  %5 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%3 : tensor<64x512xf32>) outs(%4 : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %v = arith.addf %in, %out : f32
    linalg.yield %v : f32
  } -> tensor<64xf32>
  return %5 : tensor<64xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: func.func @neg_elementwise_folded_into_r2
// The loop is unchanged: it carries a single accumulator and holds only R1.
// CHECK:         scf.for %{{.+}} = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%{{.+}} = %{{.+}}) -> (tensor<64xf32>) {
// CHECK:           linalg.generic
// CHECK:             arith.maximumf
// CHECK-NOT:       linalg.generic
// CHECK:         }
// E and R2 remain outside the loop.
// CHECK:         linalg.generic
func.func @neg_elementwise_folded_into_r2(%arg0: tensor<64x512xf32>) -> tensor<64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %cstn = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %c32 = arith.constant 32 : index
  %0 = tensor.empty() : tensor<64xf32>
  %1 = linalg.fill ins(%cstn : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %2 = scf.for %a = %c0 to %c512 step %c32 iter_args(%b = %1) -> (tensor<64xf32>) {
    %es = tensor.extract_slice %arg0[0, %a] [64, 32] [1, 1] : tensor<64x512xf32> to tensor<64x32xf32>
    %es1 = tensor.extract_slice %b[0] [64] [1] : tensor<64xf32> to tensor<64xf32>
    %m = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%es : tensor<64x32xf32>) outs(%es1 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %v = arith.maximumf %in, %out : f32
      linalg.yield %v : f32
    } -> tensor<64xf32>
    %ins = tensor.insert_slice %m into %b[0] [64] [1] : tensor<64xf32> into tensor<64xf32>
    scf.yield %ins : tensor<64xf32>
  } {__reduction_loop__}
  %3 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %4 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0, %2 : tensor<64x512xf32>, tensor<64xf32>) outs(%3 : tensor<64xf32>) {
  ^bb0(%in: f32, %mv: f32, %out: f32):
    %d = arith.subf %in, %mv : f32
    %e = math.exp %d : f32
    %v = arith.addf %e, %out : f32
    linalg.yield %v : f32
  } -> tensor<64xf32>
  return %4 : tensor<64xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: func.func @neg_r2_not_add_combiner
// The loop is unchanged: it carries a single accumulator and holds only R1.
// CHECK:         scf.for %{{.+}} = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%{{.+}} = %{{.+}}) -> (tensor<64xf32>) {
// CHECK:           linalg.generic
// CHECK:             arith.maximumf
// CHECK-NOT:       linalg.generic
// CHECK:         }
// E and R2 remain outside the loop.
// CHECK:         linalg.generic
func.func @neg_r2_not_add_combiner(%arg0: tensor<64x512xf32>, %argd: tensor<64x512xf32>) -> tensor<64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %cstn = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %c32 = arith.constant 32 : index
  %0 = tensor.empty() : tensor<64xf32>
  %1 = linalg.fill ins(%cstn : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %2 = scf.for %a = %c0 to %c512 step %c32 iter_args(%b = %1) -> (tensor<64xf32>) {
    %es = tensor.extract_slice %arg0[0, %a] [64, 32] [1, 1] : tensor<64x512xf32> to tensor<64x32xf32>
    %es1 = tensor.extract_slice %b[0] [64] [1] : tensor<64xf32> to tensor<64xf32>
    %m = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%es : tensor<64x32xf32>) outs(%es1 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %v = arith.maximumf %in, %out : f32
      linalg.yield %v : f32
    } -> tensor<64xf32>
    %ins = tensor.insert_slice %m into %b[0] [64] [1] : tensor<64xf32> into tensor<64xf32>
    scf.yield %ins : tensor<64xf32>
  } {__reduction_loop__}
  %3 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %2 : tensor<64x512xf32>, tensor<64xf32>) outs(%argd : tensor<64x512xf32>) {
  ^bb0(%in: f32, %mv: f32, %out: f32):
    %d = arith.subf %in, %mv : f32
    %v = math.exp %d : f32
    linalg.yield %v : f32
  } -> tensor<64x512xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %5 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%3 : tensor<64x512xf32>) outs(%4 : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %v = arith.maximumf %in, %out : f32
    linalg.yield %v : f32
  } -> tensor<64xf32>
  return %5 : tensor<64xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: func.func @neg_r2_init_not_zero
// The loop is unchanged: it carries a single accumulator and holds only R1.
// CHECK:         scf.for %{{.+}} = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%{{.+}} = %{{.+}}) -> (tensor<64xf32>) {
// CHECK:           linalg.generic
// CHECK:             arith.maximumf
// CHECK-NOT:       linalg.generic
// CHECK:         }
// E and R2 remain outside the loop.
// CHECK:         linalg.generic
func.func @neg_r2_init_not_zero(%arg0: tensor<64x512xf32>, %argd: tensor<64x512xf32>) -> tensor<64xf32> {
  %one = arith.constant 1.000000e+00 : f32
  %cstn = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %c32 = arith.constant 32 : index
  %0 = tensor.empty() : tensor<64xf32>
  %1 = linalg.fill ins(%cstn : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %2 = scf.for %a = %c0 to %c512 step %c32 iter_args(%b = %1) -> (tensor<64xf32>) {
    %es = tensor.extract_slice %arg0[0, %a] [64, 32] [1, 1] : tensor<64x512xf32> to tensor<64x32xf32>
    %es1 = tensor.extract_slice %b[0] [64] [1] : tensor<64xf32> to tensor<64xf32>
    %m = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%es : tensor<64x32xf32>) outs(%es1 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %v = arith.maximumf %in, %out : f32
      linalg.yield %v : f32
    } -> tensor<64xf32>
    %ins = tensor.insert_slice %m into %b[0] [64] [1] : tensor<64xf32> into tensor<64xf32>
    scf.yield %ins : tensor<64xf32>
  } {__reduction_loop__}
  %3 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %2 : tensor<64x512xf32>, tensor<64xf32>) outs(%argd : tensor<64x512xf32>) {
  ^bb0(%in: f32, %mv: f32, %out: f32):
    %d = arith.subf %in, %mv : f32
    %v = math.exp %d : f32
    linalg.yield %v : f32
  } -> tensor<64x512xf32>
  %4 = linalg.fill ins(%one : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %5 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%3 : tensor<64x512xf32>) outs(%4 : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %v = arith.addf %in, %out : f32
    linalg.yield %v : f32
  } -> tensor<64xf32>
  return %5 : tensor<64xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: func.func @neg_loop_not_annotated
// The loop is unchanged: it carries a single accumulator and holds only R1.
// CHECK:         scf.for %{{.+}} = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%{{.+}} = %{{.+}}) -> (tensor<64xf32>) {
// CHECK:           linalg.generic
// CHECK:             arith.maximumf
// CHECK-NOT:       linalg.generic
// CHECK:         }
// E and R2 remain outside the loop.
// CHECK:         linalg.generic
func.func @neg_loop_not_annotated(%arg0: tensor<64x512xf32>, %argd: tensor<64x512xf32>) -> tensor<64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %cstn = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %c32 = arith.constant 32 : index
  %0 = tensor.empty() : tensor<64xf32>
  %1 = linalg.fill ins(%cstn : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %2 = scf.for %a = %c0 to %c512 step %c32 iter_args(%b = %1) -> (tensor<64xf32>) {
    %es = tensor.extract_slice %arg0[0, %a] [64, 32] [1, 1] : tensor<64x512xf32> to tensor<64x32xf32>
    %es1 = tensor.extract_slice %b[0] [64] [1] : tensor<64xf32> to tensor<64xf32>
    %m = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%es : tensor<64x32xf32>) outs(%es1 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %v = arith.maximumf %in, %out : f32
      linalg.yield %v : f32
    } -> tensor<64xf32>
    %ins = tensor.insert_slice %m into %b[0] [64] [1] : tensor<64xf32> into tensor<64xf32>
    scf.yield %ins : tensor<64xf32>
  }
  %3 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %2 : tensor<64x512xf32>, tensor<64xf32>) outs(%argd : tensor<64x512xf32>) {
  ^bb0(%in: f32, %mv: f32, %out: f32):
    %d = arith.subf %in, %mv : f32
    %v = math.exp %d : f32
    linalg.yield %v : f32
  } -> tensor<64x512xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %5 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%3 : tensor<64x512xf32>) outs(%4 : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %v = arith.addf %in, %out : f32
    linalg.yield %v : f32
  } -> tensor<64xf32>
  return %5 : tensor<64xf32>
}

// -----

#mapik = affine_map<(d0, d1, d2) -> (d0, d2)>
#mapkj = affine_map<(d0, d1, d2) -> (d2, d1)>
#mapij = affine_map<(d0, d1, d2) -> (d0, d1)>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: func.func @r2_gemm_all_inputs_reduced
// CHECK-SAME:      (%[[X:.+]]: tensor<64x512xf32>, %[[V:.+]]: tensor<512x128xf32>, %[[DST:.+]]: tensor<64x512xf32>)
// CHECK-DAG:     %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:     %[[NEGINF:.+]] = arith.constant 0xFF800000 : f32
// CHECK:         %[[MINIT:.+]] = linalg.fill ins(%[[NEGINF]]
// The GEMM accumulator init is hoisted above the loop.
// CHECK:         %[[ACCINIT:.+]] = linalg.fill ins(%[[ZERO]]{{.*}}tensor<64x128xf32>
// CHECK:         %[[LOOP:.+]]:3 = scf.for %[[IV:[a-zA-Z0-9_]+]] =
// CHECK-SAME:        iter_args(%[[MARG:.+]] = %[[MINIT]], %[[EARG:.+]] = %[[DST]], %[[ACCARG:.+]] = %[[ACCINIT]])
// CHECK-SAME:        -> (tensor<64xf32>, tensor<64x512xf32>, tensor<64x128xf32>)
// CHECK:           %[[XTILE:.+]] = tensor.extract_slice %[[X]][0, %[[IV]]] [64, 32] [1, 1]
// CHECK:           %[[MOLD:.+]] = tensor.extract_slice %[[MARG]][0] [64] [1]
// CHECK:           %[[M:.+]] = linalg.generic
// CHECK-SAME:          ins(%[[XTILE]] : tensor<64x32xf32>)
// CHECK-SAME:          outs(%[[MOLD]] : tensor<64xf32>)
// CHECK:             arith.maximumf
// CHECK:           %[[E:.+]] = linalg.generic
// CHECK-SAME:          tensor<64x32xf32>, tensor<64xf32>
// CHECK:             arith.subf
// CHECK:             math.exp
// The contraction is fused; `v` is re-sliced along the reduction (k) axis while
// the [64x128] accumulator, which does not span k, keeps its full shape.
// CHECK:           %[[VSLICE:.+]] = tensor.extract_slice %[[V]][%[[IV]], 0] [32, 128] [1, 1]
// CHECK:           %[[ACCSLICE:.+]] = tensor.extract_slice %[[ACCARG]][0, 0] [64, 128] [1, 1]
// The factor is built over R2's parallel space, so `m` broadcasts over the extra
// N dim and each term is [64x128] rather than [64].
// CHECK:           %[[TERMNEW:.+]] = linalg.generic
// CHECK-SAME:          ins(%[[M]] : tensor<64xf32>)
// CHECK-SAME:          outs(%{{.+}} : tensor<64x128xf32>)
// CHECK:             math.exp
// CHECK:           %[[TERMOLD:.+]] = linalg.generic
// CHECK-SAME:          ins(%[[MOLD]] : tensor<64xf32>)
// CHECK-SAME:          outs(%{{.+}} : tensor<64x128xf32>)
// CHECK:             math.exp
// CHECK:           %[[FACTOR:.+]] = linalg.elementwise kind=#linalg.elementwise_kind<div> ins(%[[TERMNEW]], %[[TERMOLD]] : tensor<64x128xf32>, tensor<64x128xf32>)
// CHECK:           %[[SCALED:.+]] = linalg.elementwise kind=#linalg.elementwise_kind<mul> ins(%[[ACCSLICE]], %[[FACTOR]] : tensor<64x128xf32>, tensor<64x128xf32>)
// CHECK:           %[[ACC:.+]] = linalg.generic
// CHECK-SAME:          ins(%[[E]], %[[VSLICE]] : tensor<64x32xf32>, tensor<32x128xf32>)
// CHECK-SAME:          outs(%[[SCALED]] : tensor<64x128xf32>)
// CHECK:             arith.mulf
// CHECK:             arith.addf
// CHECK:           scf.yield
// CHECK:         return %[[LOOP]]#2
func.func @r2_gemm_all_inputs_reduced(%arg0: tensor<64x512xf32>, %v: tensor<512x128xf32>, %argd: tensor<64x512xf32>) -> tensor<64x128xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %cstn = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %c32 = arith.constant 32 : index
  %0 = tensor.empty() : tensor<64xf32>
  %1 = linalg.fill ins(%cstn : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %2 = scf.for %a = %c0 to %c512 step %c32 iter_args(%b = %1) -> (tensor<64xf32>) {
    %es = tensor.extract_slice %arg0[0, %a] [64, 32] [1, 1] : tensor<64x512xf32> to tensor<64x32xf32>
    %es1 = tensor.extract_slice %b[0] [64] [1] : tensor<64xf32> to tensor<64xf32>
    %m = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%es : tensor<64x32xf32>) outs(%es1 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %x = arith.maximumf %in, %out : f32
      linalg.yield %x : f32
    } -> tensor<64xf32>
    %ins = tensor.insert_slice %m into %b[0] [64] [1] : tensor<64xf32> into tensor<64xf32>
    scf.yield %ins : tensor<64xf32>
  } {__reduction_loop__}
  %3 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %2 : tensor<64x512xf32>, tensor<64xf32>) outs(%argd : tensor<64x512xf32>) {
  ^bb0(%in: f32, %mv: f32, %out: f32):
    %d = arith.subf %in, %mv : f32
    %x = math.exp %d : f32
    linalg.yield %x : f32
  } -> tensor<64x512xf32>
  %o = tensor.empty() : tensor<64x128xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%o : tensor<64x128xf32>) -> tensor<64x128xf32>
  %5 = linalg.generic {indexing_maps = [#mapik, #mapkj, #mapij], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %v : tensor<64x512xf32>, tensor<512x128xf32>) outs(%4 : tensor<64x128xf32>) {
  ^bb0(%p: f32, %vv: f32, %out: f32):
    %pv = arith.mulf %p, %vv : f32
    %acc = arith.addf %pv, %out : f32
    linalg.yield %acc : f32
  } -> tensor<64x128xf32>
  return %5 : tensor<64x128xf32>
}

// -----

#mapik = affine_map<(d0, d1, d2) -> (d0, d2)>
#mapbc = affine_map<(d0, d1, d2) -> (d1)>
#mapij = affine_map<(d0, d1, d2) -> (d0, d1)>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: func.func @neg_r2_input_not_reduced
// The loop is unchanged: it carries a single accumulator and holds only R1.
// CHECK:         scf.for %{{.+}} = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%{{.+}} = %{{.+}}) -> (tensor<64xf32>) {
// CHECK:           linalg.generic
// CHECK:             arith.maximumf
// CHECK-NOT:       linalg.generic
// CHECK:         }
// E and R2 remain outside the loop.
// CHECK:         linalg.generic
func.func @neg_r2_input_not_reduced(%arg0: tensor<64x512xf32>, %v: tensor<128xf32>, %argd: tensor<64x512xf32>) -> tensor<64x128xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %cstn = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %c32 = arith.constant 32 : index
  %0 = tensor.empty() : tensor<64xf32>
  %1 = linalg.fill ins(%cstn : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %2 = scf.for %a = %c0 to %c512 step %c32 iter_args(%b = %1) -> (tensor<64xf32>) {
    %es = tensor.extract_slice %arg0[0, %a] [64, 32] [1, 1] : tensor<64x512xf32> to tensor<64x32xf32>
    %es1 = tensor.extract_slice %b[0] [64] [1] : tensor<64xf32> to tensor<64xf32>
    %m = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%es : tensor<64x32xf32>) outs(%es1 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %x = arith.maximumf %in, %out : f32
      linalg.yield %x : f32
    } -> tensor<64xf32>
    %ins = tensor.insert_slice %m into %b[0] [64] [1] : tensor<64xf32> into tensor<64xf32>
    scf.yield %ins : tensor<64xf32>
  } {__reduction_loop__}
  %3 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %2 : tensor<64x512xf32>, tensor<64xf32>) outs(%argd : tensor<64x512xf32>) {
  ^bb0(%in: f32, %mv: f32, %out: f32):
    %d = arith.subf %in, %mv : f32
    %x = math.exp %d : f32
    linalg.yield %x : f32
  } -> tensor<64x512xf32>
  %o = tensor.empty() : tensor<64x128xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%o : tensor<64x128xf32>) -> tensor<64x128xf32>
  %5 = linalg.generic {indexing_maps = [#mapik, #mapbc, #mapij], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %v : tensor<64x512xf32>, tensor<128xf32>) outs(%4 : tensor<64x128xf32>) {
  ^bb0(%p: f32, %vv: f32, %out: f32):
    %pv = arith.mulf %p, %vv : f32
    %acc = arith.addf %pv, %out : f32
    linalg.yield %acc : f32
  } -> tensor<64x128xf32>
  return %5 : tensor<64x128xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

// A variance-shaped term `E = (x - m)^2`. Every other condition holds -- R1 is a
// tiled max, R2 is a zero-initialized sum -- but `E` is not multiplicatively
// separable in `m`: `x - m` carries only an additive dependence and there is no
// `exp` to convert it, so the `mulf` has no factorization to preserve. No
// per-slice scalar can rescale a stale accumulator here, so the chain is
// rejected rather than silently miscompiled.

// CHECK-LABEL: func.func @neg_e_not_separable_variance
// The loop is unchanged: it carries a single accumulator and holds only R1.
// CHECK:         scf.for %{{.+}} = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%{{.+}} = %{{.+}}) -> (tensor<64xf32>) {
// CHECK:           linalg.generic
// CHECK:             arith.maximumf
// CHECK-NOT:       linalg.generic
// CHECK:         }
// E and R2 remain outside the loop.
// CHECK:         linalg.generic
// CHECK:           arith.subf
// CHECK:           arith.mulf
func.func @neg_e_not_separable_variance(%arg0: tensor<64x512xf32>, %argd: tensor<64x512xf32>) -> tensor<64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %cstn = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %c32 = arith.constant 32 : index
  %0 = tensor.empty() : tensor<64xf32>
  %1 = linalg.fill ins(%cstn : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %2 = scf.for %a = %c0 to %c512 step %c32 iter_args(%b = %1) -> (tensor<64xf32>) {
    %es = tensor.extract_slice %arg0[0, %a] [64, 32] [1, 1] : tensor<64x512xf32> to tensor<64x32xf32>
    %es1 = tensor.extract_slice %b[0] [64] [1] : tensor<64xf32> to tensor<64xf32>
    %m = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%es : tensor<64x32xf32>) outs(%es1 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %v = arith.maximumf %in, %out : f32
      linalg.yield %v : f32
    } -> tensor<64xf32>
    %ins = tensor.insert_slice %m into %b[0] [64] [1] : tensor<64xf32> into tensor<64xf32>
    scf.yield %ins : tensor<64xf32>
  } {__reduction_loop__}
  %3 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %2 : tensor<64x512xf32>, tensor<64xf32>) outs(%argd : tensor<64x512xf32>) {
  ^bb0(%in: f32, %mv: f32, %out: f32):
    %d = arith.subf %in, %mv : f32
    %v = arith.mulf %d, %d : f32
    linalg.yield %v : f32
  } -> tensor<64x512xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %5 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%3 : tensor<64x512xf32>) outs(%4 : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %v = arith.addf %in, %out : f32
    linalg.yield %v : f32
  } -> tensor<64xf32>
  return %5 : tensor<64xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

// An L1-norm term `E = abs(x / m)` reaches multiplicative separability without
// passing through an `exp`: the `divf` gives `g(m) = 1/m` directly and `absf`
// preserves it, since `|g * h| = |g| * |h|`. The correction is therefore
// `|1/m_new| / |1/m_old|`.

// CHECK-LABEL: func.func @e_separable_abs_of_divide
// CHECK-DAG:     %[[ONE:.+]] = arith.constant 1.000000e+00 : f32
// The loop carries `m`, the fused E clone's full result and `s`.
// CHECK:         scf.for %[[IV:[a-zA-Z0-9_]+]] =
// CHECK-SAME:        -> (tensor<64xf32>, tensor<64x512xf32>, tensor<64xf32>)
// R1's DPS init holds the previous running `m`; the "old" term reads it.
// CHECK:           %[[MOLD:.+]] = tensor.extract_slice %{{.+}}[0] [64] [1]
// CHECK:           %[[M:.+]] = linalg.generic
// CHECK:             arith.maximumf
// E's clone is fused and re-sliced to the tile.
// CHECK:           %[[E:.+]] = linalg.generic
// CHECK:             arith.divf
// CHECK:             math.absf
// CHECK:           } -> tensor<64x32xf32>
// The factor is `|1/m_new| / |1/m_old|`: the data operand is neutralized to 1.0
// at the `divf` consuming it, and `absf` carries the accumulator factor through.
// CHECK:           %[[TERMNEW:.+]] = linalg.generic
// CHECK-SAME:          ins(%[[M]] : tensor<64xf32>)
// CHECK:             arith.divf %[[ONE]]
// CHECK:             math.absf
// CHECK:           %[[TERMOLD:.+]] = linalg.generic
// CHECK-SAME:          ins(%[[MOLD]] : tensor<64xf32>)
// CHECK:             arith.divf %[[ONE]]
// CHECK:             math.absf
// CHECK:           %[[FACTOR:.+]] = linalg.elementwise kind=#linalg.elementwise_kind<div> ins(%[[TERMNEW]], %[[TERMOLD]] : tensor<64xf32>, tensor<64xf32>)
// CHECK:           %[[SCALED:.+]] = linalg.elementwise kind=#linalg.elementwise_kind<mul> ins(%{{.+}}, %[[FACTOR]] : tensor<64xf32>, tensor<64xf32>)
// CHECK:           linalg.generic
// CHECK-SAME:          ins(%[[E]] : tensor<64x32xf32>)
// CHECK-SAME:          outs(%[[SCALED]] : tensor<64xf32>)
// CHECK:             arith.addf
func.func @e_separable_abs_of_divide(%arg0: tensor<64x512xf32>, %argd: tensor<64x512xf32>) -> tensor<64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %c32 = arith.constant 32 : index
  %0 = tensor.empty() : tensor<64xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %2 = scf.for %a = %c0 to %c512 step %c32 iter_args(%b = %1) -> (tensor<64xf32>) {
    %es = tensor.extract_slice %arg0[0, %a] [64, 32] [1, 1] : tensor<64x512xf32> to tensor<64x32xf32>
    %es1 = tensor.extract_slice %b[0] [64] [1] : tensor<64xf32> to tensor<64xf32>
    %m = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%es : tensor<64x32xf32>) outs(%es1 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %av = math.absf %in : f32
      %v = arith.maximumf %av, %out : f32
      linalg.yield %v : f32
    } -> tensor<64xf32>
    %ins = tensor.insert_slice %m into %b[0] [64] [1] : tensor<64xf32> into tensor<64xf32>
    scf.yield %ins : tensor<64xf32>
  } {__reduction_loop__}
  %3 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %2 : tensor<64x512xf32>, tensor<64xf32>) outs(%argd : tensor<64x512xf32>) {
  ^bb0(%in: f32, %mv: f32, %out: f32):
    %d = arith.divf %in, %mv : f32
    %v = math.absf %d : f32
    linalg.yield %v : f32
  } -> tensor<64x512xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %5 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%3 : tensor<64x512xf32>) outs(%4 : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %v = arith.addf %in, %out : f32
    linalg.yield %v : f32
  } -> tensor<64xf32>
  return %5 : tensor<64xf32>
}
