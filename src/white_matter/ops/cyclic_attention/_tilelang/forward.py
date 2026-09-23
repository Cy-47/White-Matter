"""Cyclic causal attention; Q/output use BQHD, K/V use BHSD.

Softmax uses exp2 with scale log2(e)/sqrt(D). Save natural-log LSE as
max/sqrt(D) + log(sum_exp) so backward can reconstruct probabilities.
Residue is a runtime device scalar shared by all same-shaped cyclic groups.
"""

import tilelang.language as T

_TL_DTYPE = "bfloat16"
_ACCUM_DTYPE = "float32"
_LSE_DTYPE = "float32"


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
    key_strides=None,
    value_strides=None,
):
    groups = HQ // HKV
    inv_sqrt_d = (1.0 / D) ** 0.5
    scale = inv_sqrt_d * 1.44269504  # scale / ln(2) for exp2 path

    q_shape = [B, Q_LEN, HQ, D]
    kv_shape = [B, HKV, T_kv, D]
    key_strides = key_strides or (HKV * T_kv * D, T_kv * D, D, 1)
    value_strides = value_strides or (HKV * T_kv * D, T_kv * D, D, 1)
    lse_shape = [B, HQ, Q_LEN]

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, _TL_DTYPE),
        K: T.StridedTensor(kv_shape, key_strides, _TL_DTYPE),
        V: T.StridedTensor(kv_shape, value_strides, _TL_DTYPE),
        Residue: T.Tensor([1], "int32"),
        Output: T.Tensor(q_shape, _TL_DTYPE),
        Lse: T.Tensor(lse_shape, _LSE_DTYPE),
    ):
        with T.Kernel(T.ceildiv(Q_LEN * groups, block_M), HKV, B, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, D], _TL_DTYPE)
            K_shared = T.alloc_shared([block_N, D], _TL_DTYPE)
            V_shared = T.alloc_shared([block_N, D], _TL_DTYPE)
            O_shared = T.alloc_shared([block_M, D], _TL_DTYPE)
            acc_s = T.alloc_fragment([block_M, block_N], _ACCUM_DTYPE)
            acc_s_cast = T.alloc_fragment([block_M, block_N], _TL_DTYPE)
            acc_o = T.alloc_fragment([block_M, D], _ACCUM_DTYPE)
            scores_max = T.alloc_fragment([block_M], _ACCUM_DTYPE)
            scores_max_prev = T.alloc_fragment([block_M], _ACCUM_DTYPE)
            scores_scale = T.alloc_fragment([block_M], _ACCUM_DTYPE)
            scores_sum = T.alloc_fragment([block_M], _ACCUM_DTYPE)
            logsum = T.alloc_fragment([block_M], _ACCUM_DTYPE)

            # Pack query heads sharing a KV head into the same attention tile.
            for i, d in T.Parallel(block_M, D):
                Q_shared[i, d] = Q[bz, (bx * block_M + i) // groups, by * groups + (bx * block_M + i) % groups, d]
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(_ACCUM_DTYPE))

            residue = Residue[0]
            q_actual_max = residue + (((bx + 1) * block_M - 1) // groups) * K_stride
            loop_range = T.min(
                T.ceildiv(T_kv, block_N),
                T.ceildiv(q_actual_max + 1, block_N),
            )

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(K[bz, by, k * block_N : (k + 1) * block_N, :], K_shared)
                for i, j in T.Parallel(block_M, block_N):
                    q_actual = residue + ((bx * block_M + i) // groups) * K_stride
                    acc_s[i, j] = T.if_then_else(
                        q_actual >= k * block_N + j,
                        0,
                        -T.infinity(acc_s.dtype),
                    )
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    # An unchanged maximum must rescale by exactly one.
                    scores_scale[i] = T.exp2((scores_max_prev[i] - scores_max[i]) * scale)
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)

                for i, j in T.Parallel(block_M, D):
                    acc_o[i, j] *= scores_scale[i]

                T.copy(V[bz, by, k * block_N : (k + 1) * block_N, :], V_shared)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(block_M, D):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            for i, d in T.Parallel(block_M, D):
                Output[bz, (bx * block_M + i) // groups, by * groups + (bx * block_M + i) % groups, d] = O_shared[i, d]

            # Natural-log LSE for backward probability reconstruction.
            for i in T.Parallel(block_M):
                Lse[bz, by * groups + (bx * block_M + i) % groups, (bx * block_M + i) // groups] = scores_max[i] * inv_sqrt_d + T.log(logsum[i])

    return main
