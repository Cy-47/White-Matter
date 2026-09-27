"""Document-masked cyclic dK/dV kernel with a document-derived Q-loop bound."""

import tilelang.language as T


def build_program(
    B,
    T_kv,
    HQ,
    HKV,
    D,
    q_len,
    K_stride,
    block_M,
    block_N,
    num_stages,
    threads,
):
    groups = HQ // HKV
    inv_sqrt_d = (1.0 / D) ** 0.5
    scale_log2 = inv_sqrt_d * 1.44269504
    n_q_blocks = (q_len + block_M - 1) // block_M
    q_shape = [B, HQ, q_len, D]
    kv_shape = [B, HKV, T_kv, D]
    row_shape = [B, HQ, q_len]

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, "bfloat16"),
        K: T.Tensor(kv_shape, "bfloat16"),
        V: T.Tensor(kv_shape, "bfloat16"),
        dO: T.Tensor(q_shape, "bfloat16"),
        D_pre: T.Tensor(row_shape, "float32"),
        Lse: T.Tensor(row_shape, "float32"),
        QSeg: T.Tensor([B, q_len], "int32"),
        KSeg: T.Tensor([B, T_kv], "int32"),
        KQEnd: T.Tensor([B, T_kv], "int32"),
        Residue: T.Tensor([1], "int32"),
        dK: T.Tensor(kv_shape, "bfloat16"),
        dV: T.Tensor(kv_shape, "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(T_kv, block_N), HKV, B, threads=threads) as (bx, by, bz):
            k_shared = T.alloc_shared([block_N, D], "bfloat16")
            v_shared = T.alloc_shared([block_N, D], "bfloat16")
            q_shared = T.alloc_shared([block_M, D], "bfloat16")
            do_shared = T.alloc_shared([block_M, D], "bfloat16")
            acc_s = T.alloc_fragment([block_M, block_N], "float32")
            p_shared = T.alloc_shared([block_M, block_N], "bfloat16")
            ds_shared = T.alloc_shared([block_M, block_N], "bfloat16")
            acc_dp = T.alloc_fragment([block_M, block_N], "float32")
            acc_dk = T.alloc_fragment([block_N, D], "float32")
            acc_dv = T.alloc_fragment([block_N, D], "float32")
            d_row = T.alloc_fragment([block_M], "float32")
            l_row = T.alloc_fragment([block_M], "float32")
            q_seg = T.alloc_fragment([block_M], "int32")
            k_seg = T.alloc_fragment([block_N], "int32")
            dk_shared = T.alloc_shared([block_N, D], "bfloat16")
            dv_shared = T.alloc_shared([block_N, D], "bfloat16")

            T.copy(K[bz, by, bx * block_N : (bx + 1) * block_N, :], k_shared)
            T.copy(V[bz, by, bx * block_N : (bx + 1) * block_N, :], v_shared)
            for j in T.Parallel(block_N):
                k_seg[j] = KSeg[bz, T.min(bx * block_N + j, T_kv - 1)]
            T.fill(acc_dk, 0.0)
            T.fill(acc_dv, 0.0)

            residue = Residue[0]
            i_blk_start = T.max(0, (bx * block_N - residue) // (K_stride * block_M))
            # KQEnd is the exclusive query-index end of each key's document.
            # Taking the final key in the slab covers every document touched by
            # that slab. Slab zero contains the globally-visible dummy and is
            # assigned q_len by the caller.
            q_end = T.if_then_else(
                bx == 0,
                q_len,
                KQEnd[bz, T.min((bx + 1) * block_N - 1, T_kv - 1)],
            )
            i_blk_end = T.min(n_q_blocks, T.ceildiv(q_end, block_M))

            # Slabs invisible to every real query retain their zero accumulators.
            if bx * block_N <= residue + (q_len - 1) * K_stride:
                for g in T.serial(groups):
                    hq = by * groups + g
                    for ib in T.Pipelined(i_blk_start, i_blk_end, num_stages=num_stages):
                        T.copy(
                            Q[bz, hq, ib * block_M : (ib + 1) * block_M, :],
                            q_shared,
                        )
                        T.copy(
                            dO[bz, hq, ib * block_M : (ib + 1) * block_M, :],
                            do_shared,
                        )
                        for i in T.Parallel(block_M):
                            d_row[i] = T.if_then_else(ib * block_M + i < q_len, D_pre[bz, hq, ib * block_M + i], 0.0)
                            l_row[i] = T.if_then_else(
                                ib * block_M + i < q_len, Lse[bz, hq, ib * block_M + i] * 1.44269504, 0.0
                            )
                            q_seg[i] = QSeg[bz, T.min(ib * block_M + i, q_len - 1)]

                        T.fill(acc_s, 0.0)
                        T.gemm(
                            q_shared,
                            k_shared,
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        for i, j in T.Parallel(block_M, block_N):
                            q_actual = residue + (ib * block_M + i) * K_stride
                            acc_s[i, j] = T.if_then_else(
                                (ib * block_M + i < q_len)
                                and (bx * block_N + j < T_kv)
                                and (q_actual >= bx * block_N + j),
                                T.if_then_else(
                                    q_seg[i] == k_seg[j],
                                    T.exp2(acc_s[i, j] * scale_log2 - l_row[i]),
                                    T.if_then_else(
                                        k_seg[j] == -1,
                                        T.exp2(acc_s[i, j] * scale_log2 - l_row[i]),
                                        0.0,
                                    ),
                                ),
                                0.0,
                            )
                        T.copy(acc_s, p_shared)
                        T.gemm(
                            p_shared,
                            do_shared,
                            acc_dv,
                            transpose_A=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        T.fill(acc_dp, 0.0)
                        T.gemm(
                            do_shared,
                            v_shared,
                            acc_dp,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] *= acc_dp[i, j] - d_row[i]
                        T.copy(acc_s, ds_shared)
                        T.gemm(
                            ds_shared,
                            q_shared,
                            acc_dk,
                            transpose_A=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )

            for j, d in T.Parallel(block_N, D):
                acc_dk[j, d] *= inv_sqrt_d
            T.copy(acc_dk, dk_shared)
            T.copy(acc_dv, dv_shared)
            T.copy(
                dk_shared,
                dK[bz, by, bx * block_N : (bx + 1) * block_N, :],
            )
            T.copy(
                dv_shared,
                dV[bz, by, bx * block_N : (bx + 1) * block_N, :],
            )

    return main
