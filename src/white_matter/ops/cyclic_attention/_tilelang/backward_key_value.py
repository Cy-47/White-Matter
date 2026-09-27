"""KV-parallel cyclic backward, accumulating dK/dV across grouped query heads.

Each KV tile starts at its earliest potentially visible query tile. P and dS
use shared-memory bridges to support the transposed GEMMs without fragment
layout conflicts. Outputs have the same BHSD layout as K/V.
"""

import tilelang.language as T

_TL_DTYPE = "bfloat16"
_ACCUM_DTYPE = "float32"
_LSE_DTYPE = "float32"
_LOG2E = 1.44269504


def build_program(
    B,
    T_kv,
    HQ,
    HKV,
    D,
    Q_LEN,
    K_stride,
    block_M,
    block_N,
    num_stages,
    threads,
):
    groups = HQ // HKV
    inv_sqrt_d = (1.0 / D) ** 0.5
    scale_log2 = inv_sqrt_d * _LOG2E
    n_q_blocks = (Q_LEN + block_M - 1) // block_M

    q_shape = [B, HQ, Q_LEN, D]
    kv_shape = [B, HKV, T_kv, D]
    lse_shape = [B, HQ, Q_LEN]

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, _TL_DTYPE),
        K: T.Tensor(kv_shape, _TL_DTYPE),
        V: T.Tensor(kv_shape, _TL_DTYPE),
        dO: T.Tensor(q_shape, _TL_DTYPE),
        D_pre: T.Tensor(lse_shape, _LSE_DTYPE),  # (dO * O).sum(-1) precomputed
        Lse: T.Tensor(lse_shape, _LSE_DTYPE),
        Residue: T.Tensor([1], "int32"),
        dK: T.Tensor(kv_shape, _TL_DTYPE),
        dV: T.Tensor(kv_shape, _TL_DTYPE),
    ):
        # bx = KV-slab index, by = KV head index, bz = batch index.
        with T.Kernel(T.ceildiv(T_kv, block_N), HKV, B, threads=threads) as (bx, by, bz):
            K_shared = T.alloc_shared([block_N, D], _TL_DTYPE)
            V_shared = T.alloc_shared([block_N, D], _TL_DTYPE)
            Q_shared = T.alloc_shared([block_M, D], _TL_DTYPE)
            dO_shared = T.alloc_shared([block_M, D], _TL_DTYPE)

            acc_s = T.alloc_fragment([block_M, block_N], _ACCUM_DTYPE)
            # P and dS are routed through shared memory for transposed gemm
            # inputs (avoids the fragment-layout conflict that occurs when
            # the same fragment is used as A in both non-transposed and
            # transposed gemms).
            p_shared = T.alloc_shared([block_M, block_N], _TL_DTYPE)
            ds_shared = T.alloc_shared([block_M, block_N], _TL_DTYPE)
            acc_dp = T.alloc_fragment([block_M, block_N], _ACCUM_DTYPE)
            acc_dk = T.alloc_fragment([block_N, D], _ACCUM_DTYPE)
            acc_dv = T.alloc_fragment([block_N, D], _ACCUM_DTYPE)
            D_row = T.alloc_fragment([block_M], _ACCUM_DTYPE)
            L_row = T.alloc_fragment([block_M], _ACCUM_DTYPE)
            dk_shared = T.alloc_shared([block_N, D], _TL_DTYPE)
            dv_shared = T.alloc_shared([block_N, D], _TL_DTYPE)

            T.copy(K[bz, by, bx * block_N : (bx + 1) * block_N, :], K_shared)
            T.copy(V[bz, by, bx * block_N : (bx + 1) * block_N, :], V_shared)

            T.fill(acc_dk, 0.0)
            T.fill(acc_dv, 0.0)

            # Queries before this key slab are fully masked and contribute zero
            # to dK/dV. Start at a conservative causal lower bound.
            residue = Residue[0]
            i_blk_start = T.max(0, (bx * block_N - residue) // (K_stride * block_M))

            # Slabs invisible to every real query retain their zero accumulators.
            if bx * block_N <= residue + (Q_LEN - 1) * K_stride:
                for g in T.serial(groups):
                    hq = by * groups + g
                    for i_blk in T.Pipelined(i_blk_start, n_q_blocks, num_stages=num_stages):
                        T.copy(Q[bz, hq, i_blk * block_M : (i_blk + 1) * block_M, :], Q_shared)
                        T.copy(dO[bz, hq, i_blk * block_M : (i_blk + 1) * block_M, :], dO_shared)
                        # Load the fused row reduction D[i] = sum_d dO[i,d]*O[i,d]
                        # and LSE. Hoisting D_pre out of the kernel saves
                        # block_M×D×2B of shared memory per CTA, which lets us
                        # grow block_M from 32 → 64 at d=128.
                        for i in T.Parallel(block_M):
                            D_row[i] = T.if_then_else(
                                i_blk * block_M + i < Q_LEN, D_pre[bz, hq, i_blk * block_M + i], 0.0
                            )
                            L_row[i] = T.if_then_else(
                                i_blk * block_M + i < Q_LEN, Lse[bz, hq, i_blk * block_M + i] * _LOG2E, 0.0
                            )

                        # acc_s = Q @ K^T (no scale).
                        T.fill(acc_s, 0.0)
                        T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                        # P = exp2(acc_s*scale_log2 - L_log2[i]) with causal mask
                        # (j > p[i] -> 0).
                        for i, j in T.Parallel(block_M, block_N):
                            q_actual = residue + (i_blk * block_M + i) * K_stride
                            acc_s[i, j] = T.if_then_else(
                                (i_blk * block_M + i < Q_LEN)
                                and (bx * block_N + j < T_kv)
                                and (q_actual >= bx * block_N + j),
                                T.exp2(acc_s[i, j] * scale_log2 - L_row[i]),
                                0.0,
                            )
                        T.copy(acc_s, p_shared)

                        # dV += P^T @ dO   (acc_dv : (block_N, D))
                        T.gemm(p_shared, dO_shared, acc_dv, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)

                        # dP = dO @ V^T
                        T.fill(acc_dp, 0.0)
                        T.gemm(dO_shared, V_shared, acc_dp, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                        # dS = P * (dP - D_row[i])
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = acc_s[i, j] * (acc_dp[i, j] - D_row[i])
                        T.copy(acc_s, ds_shared)
                        # dK += dS^T @ Q
                        T.gemm(ds_shared, Q_shared, acc_dk, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)

            for j, d in T.Parallel(block_N, D):
                acc_dk[j, d] *= inv_sqrt_d
            T.copy(acc_dk, dk_shared)
            T.copy(acc_dv, dv_shared)
            T.copy(dk_shared, dK[bz, by, bx * block_N : (bx + 1) * block_N, :])
            T.copy(dv_shared, dV[bz, by, bx * block_N : (bx + 1) * block_N, :])

    return main
