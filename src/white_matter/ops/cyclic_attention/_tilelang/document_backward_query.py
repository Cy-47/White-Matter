"""Document-masked cyclic dQ kernel with a document-start KV-loop bound."""

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
        QStart: T.Tensor([B, q_len], "int32"),
        Residue: T.Tensor([1], "int32"),
        dQ: T.Tensor(q_shape, "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(q_len, block_M), HQ, B, threads=threads) as (bx, by, bz):
            q_shared = T.alloc_shared([block_M, D], "bfloat16")
            do_shared = T.alloc_shared([block_M, D], "bfloat16")
            k_shared = T.alloc_shared([block_N, D], "bfloat16")
            v_shared = T.alloc_shared([block_N, D], "bfloat16")
            dq_shared = T.alloc_shared([block_M, D], "bfloat16")
            acc_s = T.alloc_fragment([block_M, block_N], "float32")
            acc_dp = T.alloc_fragment([block_M, block_N], "float32")
            # Shared bridge permits M=32 despite producer/consumer fragment
            # layout disagreement in the stock TileLang inference.
            ds_shared = T.alloc_shared([block_M, block_N], "bfloat16")
            acc_dq = T.alloc_fragment([block_M, D], "float32")
            d_row = T.alloc_fragment([block_M], "float32")
            l_row = T.alloc_fragment([block_M], "float32")
            q_seg = T.alloc_fragment([block_M], "int32")
            k_seg = T.alloc_fragment([block_N], "int32")

            T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], q_shared)
            T.copy(dO[bz, by, bx * block_M : (bx + 1) * block_M, :], do_shared)
            for i in T.Parallel(block_M):
                d_row[i] = D_pre[bz, by, bx * block_M + i]
                l_row[i] = Lse[bz, by, bx * block_M + i] * 1.44269504
                q_seg[i] = QSeg[bz, T.min(bx * block_M + i, q_len - 1)]
            T.fill(acc_dq, 0.0)

            residue = Residue[0]
            q_actual_max = residue + ((bx + 1) * block_M - 1) * K_stride
            dense_end = T.min(
                T.ceildiv(T_kv, block_N),
                T.ceildiv(q_actual_max + 1, block_N),
            )
            doc_start = QStart[bz, bx * block_M]
            main_start = T.max(1, doc_start // block_N)
            loop_range = 1 + T.max(0, dense_end - main_start)

            for kk in T.Pipelined(loop_range, num_stages=num_stages):
                k = T.if_then_else(kk == 0, 0, main_start + kk - 1)
                T.copy(
                    K[bz, by // groups, k * block_N : (k + 1) * block_N, :],
                    k_shared,
                )
                T.copy(
                    V[bz, by // groups, k * block_N : (k + 1) * block_N, :],
                    v_shared,
                )
                for j in T.Parallel(block_N):
                    k_seg[j] = KSeg[bz, T.min(k * block_N + j, T_kv - 1)]
                T.fill(acc_s, 0.0)
                T.gemm(
                    q_shared,
                    k_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                for i, j in T.Parallel(block_M, block_N):
                    q_actual = residue + (bx * block_M + i) * K_stride
                    acc_s[i, j] = T.if_then_else(
                        q_actual >= k * block_N + j,
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
                    k_shared,
                    acc_dq,
                    policy=T.GemmWarpPolicy.FullRow,
                )

            for i, d in T.Parallel(block_M, D):
                acc_dq[i, d] *= inv_sqrt_d
            T.copy(acc_dq, dq_shared)
            T.copy(
                dq_shared,
                dQ[bz, by, bx * block_M : (bx + 1) * block_M, :],
            )

    return main
