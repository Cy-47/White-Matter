"""Query-parallel cyclic backward: dQ = (P * (dO @ V.T - D_row)) @ K / sqrt(D).

D_row = sum(dO * O) is precomputed once. Each query tile visits
only causally visible K/V tiles; saved natural-log LSE reconstructs P.
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
        dQ: T.Tensor(q_shape, _TL_DTYPE),
    ):
        with T.Kernel(T.ceildiv(Q_LEN, block_M), HQ, B, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, D], _TL_DTYPE)
            dO_shared = T.alloc_shared([block_M, D], _TL_DTYPE)
            K_shared = T.alloc_shared([block_N, D], _TL_DTYPE)
            V_shared = T.alloc_shared([block_N, D], _TL_DTYPE)
            dQ_shared = T.alloc_shared([block_M, D], _TL_DTYPE)

            acc_s = T.alloc_fragment([block_M, block_N], _ACCUM_DTYPE)
            acc_dp = T.alloc_fragment([block_M, block_N], _ACCUM_DTYPE)
            ds_cast = T.alloc_fragment([block_M, block_N], _TL_DTYPE)
            acc_dq = T.alloc_fragment([block_M, D], _ACCUM_DTYPE)
            D_row = T.alloc_fragment([block_M], _ACCUM_DTYPE)
            L_row = T.alloc_fragment([block_M], _ACCUM_DTYPE)

            T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], Q_shared)
            T.copy(dO[bz, by, bx * block_M : (bx + 1) * block_M, :], dO_shared)

            # Load precomputed D[i] and L[i]. Pulling D_pre out of the kernel
            # saves block_M×D×2B shmem per CTA (no O_shared), which frees
            # budget to grow tiles at d=128.
            for i in T.Parallel(block_M):
                D_row[i] = D_pre[bz, by, bx * block_M + i]
                L_row[i] = Lse[bz, by, bx * block_M + i] * _LOG2E

            T.fill(acc_dq, 0.0)

            residue = Residue[0]
            q_actual_max = residue + ((bx + 1) * block_M - 1) * K_stride
            loop_range = T.min(
                T.ceildiv(T_kv, block_N),
                T.ceildiv(q_actual_max + 1, block_N),
            )

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(K[bz, by // groups, k * block_N : (k + 1) * block_N, :], K_shared)
                T.copy(V[bz, by // groups, k * block_N : (k + 1) * block_N, :], V_shared)

                # acc_s = Q @ K^T (no scale yet)
                T.fill(acc_s, 0.0)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                # P = exp2(acc_s * scale_log2 - L_log2) with causal mask.
                for i, j in T.Parallel(block_M, block_N):
                    q_actual = residue + (bx * block_M + i) * K_stride
                    acc_s[i, j] = T.if_then_else(
                        q_actual >= k * block_N + j,
                        T.exp2(acc_s[i, j] * scale_log2 - L_row[i]),
                        0.0,
                    )

                # dP = dO @ V^T
                T.fill(acc_dp, 0.0)
                T.gemm(dO_shared, V_shared, acc_dp, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                # dS = P * (dP - D_row[i])
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = acc_s[i, j] * (acc_dp[i, j] - D_row[i])
                T.copy(acc_s, ds_cast)

                # dQ += dS @ K * inv_sqrt_d. Fold the scale into the cast by
                # multiplying ds_cast by inv_sqrt_d? Easier: gemm then scale.
                T.gemm(ds_cast, K_shared, acc_dq, policy=T.GemmWarpPolicy.FullRow)

            for i, d in T.Parallel(block_M, D):
                acc_dq[i, d] *= inv_sqrt_d
            T.copy(acc_dq, dQ_shared)
            T.copy(dQ_shared, dQ[bz, by, bx * block_M : (bx + 1) * block_M, :])

    return main
