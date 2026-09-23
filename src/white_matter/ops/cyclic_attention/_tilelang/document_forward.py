"""Cyclic forward with document isolation and a globally visible dummy token.

Keep a key when causal and (KSeg == QSeg or KSeg == -1). Metadata uses exact
logical lengths; ragged tile loads clamp to the last valid entry. Single-document
tiles skip redundant segment comparisons. Q/output use BQHD for the projections.
LSE and runtime residue have the same convention as the plain kernel.
"""

import tilelang.language as T

_TL_DTYPE = "bfloat16"
_ACCUM_DTYPE = "float32"
_LSE_DTYPE = "float32"
_SEG_DTYPE = "int32"


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
    scale = inv_sqrt_d * 1.44269504  # scale / ln(2) for exp2 path

    q_shape = [B, Q_LEN, HQ, D]
    kv_shape = [B, HKV, T_kv, D]
    lse_shape = [B, HQ, Q_LEN]
    qseg_shape = [B, Q_LEN]
    kseg_shape = [B, T_kv]

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, _TL_DTYPE),
        K: T.Tensor(kv_shape, _TL_DTYPE),
        V: T.Tensor(kv_shape, _TL_DTYPE),
        QSeg: T.Tensor(qseg_shape, _SEG_DTYPE),
        KSeg: T.Tensor(kseg_shape, _SEG_DTYPE),
        Residue: T.Tensor([1], "int32"),
        Output: T.Tensor(q_shape, _TL_DTYPE),
        Lse: T.Tensor(lse_shape, _LSE_DTYPE),
    ):
        with T.Kernel(T.ceildiv(Q_LEN, block_M), HQ, B, threads=threads) as (bx, by, bz):
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
            q_seg = T.alloc_fragment([block_M], _SEG_DTYPE)
            k_seg = T.alloc_fragment([block_N], _SEG_DTYPE)

            T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
            # Clamp the ragged last-tile index (>= Q_LEN) to the last valid row:
            # those query rows are discarded, so the clamped seg value is unused.
            for i in T.Parallel(block_M):
                q_seg[i] = QSeg[bz, T.min(bx * block_M + i, Q_LEN - 1)]
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(_ACCUM_DTYPE))

            # Intra-doc EARLY-OUT (bit-identical): segment ids are monotonic within
            # a tile (docs are contiguous slots), so a tile is trivially intra-doc
            # iff its endpoint seg values agree. If the whole Q-block is one document
            # g, KV tiles fully inside g skip the 3-branch seg compare and use plain
            # strided-causal masking (identical keep-set). Uniform global reads ->
            # T.If branches coherently. Net win on the EOS-packed prod distribution.
            g_val = QSeg[bz, bx * block_M]
            q_last = QSeg[bz, T.min(bx * block_M + block_M - 1, Q_LEN - 1)]
            q_single = g_val == q_last

            residue = Residue[0]
            q_actual_max = residue + ((bx + 1) * block_M - 1) * K_stride
            loop_range = T.min(
                T.ceildiv(T_kv, block_N),
                T.ceildiv(q_actual_max + 1, block_N),
            )

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(K[bz, by // groups, k * block_N : (k + 1) * block_N, :], K_shared)
                # Clamp ragged kv index (>= T_kv); those keys are causally masked.
                for j in T.Parallel(block_N):
                    k_seg[j] = KSeg[bz, T.min(k * block_N + j, T_kv - 1)]
                k_first = KSeg[bz, k * block_N]
                k_last = KSeg[bz, T.min(k * block_N + block_N - 1, T_kv - 1)]
                trivial = q_single and (k_first == g_val) and (k_last == g_val)
                with T.If(trivial):
                    with T.Then():
                        for i, j in T.Parallel(block_M, block_N):
                            q_actual = residue + (bx * block_M + i) * K_stride
                            acc_s[i, j] = T.if_then_else(q_actual >= k * block_N + j, 0.0, -T.infinity(acc_s.dtype))
                    with T.Else():
                        for i, j in T.Parallel(block_M, block_N):
                            q_actual = residue + (bx * block_M + i) * K_stride
                            acc_s[i, j] = T.if_then_else(
                                q_actual >= k * block_N + j,
                                T.if_then_else(
                                    q_seg[i] == k_seg[j],
                                    0.0,
                                    T.if_then_else(k_seg[j] == -1, 0.0, -T.infinity(acc_s.dtype)),
                                ),
                                -T.infinity(acc_s.dtype),
                            )
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2((scores_max_prev[i] - scores_max[i]) * scale)
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)

                for i, j in T.Parallel(block_M, D):
                    acc_o[i, j] *= scores_scale[i]

                T.copy(V[bz, by // groups, k * block_N : (k + 1) * block_N, :], V_shared)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(block_M, D):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

            for i in T.Parallel(block_M):
                Lse[bz, by, bx * block_M + i] = scores_max[i] * inv_sqrt_d + T.log(logsum[i])

    return main
