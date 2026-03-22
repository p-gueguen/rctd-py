/**
 * Fused IRWLS inner-loop CUDA kernel.
 *
 * CUDA-Agent-inspired optimization: fuses prediction → spline interpolation →
 * gradient → hessian accumulation into a single kernel launch.
 *
 * This eliminates materializing intermediate tensors (prediction, d1_vec,
 * d2_vec, d2_w) of size N*G each, saving ~4*N*G*sizeof(double) bytes of
 * global memory traffic per IRWLS iteration.
 *
 * Uses native CUDA sqrt/floor/ceil which match PyTorch's intrinsics exactly
 * (unlike Triton's tl.math.sqrt which diverges at float64 boundaries).
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

namespace {

/**
 * Fused kernel: for each (pixel n, gene g), compute prediction, spline
 * derivatives, and accumulate into grad[n,k] and hess[n,k1*K+k2].
 *
 * Grid: (n_pixels, ceil(G/BLOCK_G))
 * Block: (BLOCK_G,) threads
 *
 * Each thread handles one (n, g) pair. Gradient and hessian are accumulated
 * via atomicAdd across genes within each pixel.
 */
template <typename scalar_t>
__global__ void fused_predict_derivatives_kernel(
    const scalar_t* __restrict__ w,        // (N, K) clamped weights
    const scalar_t* __restrict__ P,        // (G, K) reference profiles
    const scalar_t* __restrict__ P_T,      // (K, G) transposed profiles
    const scalar_t* __restrict__ Y,        // (N, G) observed counts
    const scalar_t* __restrict__ nUMI,     // (N,) UMI counts
    const scalar_t* __restrict__ thresh,   // (N,) prediction thresholds
    const scalar_t* __restrict__ Q_mat,    // (n_rows, N_X) log-likelihood table
    const scalar_t* __restrict__ SQ_mat,   // (n_rows, N_X) spline coefficients
    const scalar_t* __restrict__ x_vals,   // (N_X,) lambda grid
    scalar_t* __restrict__ grad,           // (N, K) output gradient
    scalar_t* __restrict__ hess,           // (N, K*K) output hessian
    int N, int G, int K, int N_X, int K_val,
    scalar_t X_max
) {
    int n = blockIdx.x;
    int g = blockIdx.y * blockDim.x + threadIdx.x;

    if (n >= N || g >= G) return;

    // --- Step 1: Compute prediction[n, g] = |nUMI[n] * sum_k(w[n,k] * P[g,k])| ---
    scalar_t dot = 0.0;
    for (int k = 0; k < K; k++) {
        dot += w[n * K + k] * P[g * K + k];
    }
    scalar_t numi_n = nUMI[n];
    scalar_t pred = fabs(numi_n * dot);
    scalar_t thr = thresh[n];
    if (pred < thr) pred = thr;

    // --- Step 2: Spline interpolation (inlined calc_q_all) ---
    scalar_t y_val = Y[n * G + g];
    // Clamp Y to K_val
    if (y_val > (scalar_t)K_val) y_val = (scalar_t)K_val;

    scalar_t epsilon = 1e-4;
    scalar_t lam = pred;
    if (lam < epsilon) lam = epsilon;
    if (lam > X_max - epsilon) lam = X_max - epsilon;

    scalar_t delta = 1e-6;
    // Native CUDA sqrt — matches PyTorch's torch.sqrt exactly for float64
    scalar_t l_float = floor(sqrt(lam / delta));
    long l = (long)l_float;

    // Index mapping: m = min(l-9, 40) + max(ceil(sqrt(max(l-48.7499, 0)*4))-2, 0)
    long m_part1 = l - 9;
    if (m_part1 > 40) m_part1 = 40;
    scalar_t inner = ((scalar_t)l - 48.7499);
    if (inner < 0.0) inner = 0.0;
    long m_part2 = (long)ceil(sqrt(inner * 4.0)) - 2;
    if (m_part2 < 0) m_part2 = 0;
    long m = m_part1 + m_part2;

    // Spline node values (0-indexed: m-1 and m correspond to R's m and m+1)
    long m_idx0 = m - 1;  // x_vals[m-1]
    long m_idx1 = m;      // x_vals[m]

    // Bounds check
    if (m_idx0 < 0) m_idx0 = 0;
    if (m_idx1 >= N_X) m_idx1 = N_X - 1;
    if (m_idx0 >= N_X) m_idx0 = N_X - 1;

    scalar_t ti1 = x_vals[m_idx0];
    scalar_t ti = x_vals[m_idx1];
    scalar_t hi = ti - ti1;

    long y_idx = (long)y_val;
    if (y_idx < 0) y_idx = 0;
    int n_rows = K_val + 3;
    if (y_idx >= n_rows) y_idx = n_rows - 1;

    // Table lookups
    scalar_t fti1 = Q_mat[y_idx * N_X + m_idx0];
    scalar_t fti = Q_mat[y_idx * N_X + m_idx1];
    scalar_t zi1 = SQ_mat[y_idx * N_X + m_idx0];
    scalar_t zi = SQ_mat[y_idx * N_X + m_idx1];

    scalar_t diff1 = lam - ti1;
    scalar_t diff2 = ti - lam;
    scalar_t diff3 = fti / hi - zi * hi / 6.0;
    scalar_t diff4 = fti1 / hi - zi1 * hi / 6.0;
    scalar_t zdi = zi / hi;
    scalar_t zdi1 = zi1 / hi;

    // d1 and d2 from cubic spline
    scalar_t d1 = zdi * diff1 * diff1 / 2.0 - zdi1 * diff2 * diff2 / 2.0 + diff3 - diff4;
    scalar_t d2 = zdi * diff1 + zdi1 * diff2;

    // --- Step 3: Accumulate gradient and hessian ---
    scalar_t d1_numi = d1 * numi_n;
    scalar_t neg_d2_numi2 = (-d2) * numi_n * numi_n;

    // Gradient: grad[n, k] -= d1 * nUMI * P[g, k]
    for (int k = 0; k < K; k++) {
        scalar_t p_gk = P[g * K + k];
        atomicAdd(&grad[n * K + k], -d1_numi * p_gk);
    }

    // Hessian: hess[n, k1*K+k2] += (-d2) * nUMI^2 * P[g,k1] * P[g,k2]
    for (int k1 = 0; k1 < K; k1++) {
        scalar_t p_gk1 = P[g * K + k1];
        for (int k2 = k1; k2 < K; k2++) {
            scalar_t p_gk2 = P[g * K + k2];
            scalar_t val = neg_d2_numi2 * p_gk1 * p_gk2;
            atomicAdd(&hess[n * K * K + k1 * K + k2], val);
            if (k2 != k1) {
                // Symmetric: fill lower triangle
                atomicAdd(&hess[n * K * K + k2 * K + k1], val);
            }
        }
    }
}

} // anonymous namespace


std::vector<torch::Tensor> fused_predict_and_derivatives_cuda(
    torch::Tensor w,        // (N, K) clamped weights (already clamp(min=0))
    torch::Tensor P,        // (G, K) reference profiles
    torch::Tensor P_T,      // (K, G) transposed profiles
    torch::Tensor Y,        // (N, G) observed counts
    torch::Tensor nUMI,     // (N,) UMI counts
    torch::Tensor thresh,   // (N,) thresholds
    torch::Tensor Q_mat,    // (n_rows, N_X) log-likelihood table
    torch::Tensor SQ_mat,   // (n_rows, N_X) spline coefficients
    torch::Tensor x_vals,   // (N_X,) lambda grid
    int K_val
) {
    int N = w.size(0);
    int K = w.size(1);
    int G = P.size(0);
    int N_X = x_vals.size(0);

    auto opts = torch::TensorOptions().dtype(w.dtype()).device(w.device());
    auto grad = torch::zeros({N, K}, opts);
    auto hess = torch::zeros({N, K * K}, opts);

    if (N == 0) {
        return {grad, hess.reshape({N, K, K})};
    }

    // Ensure contiguous layout
    w = w.contiguous();
    P = P.contiguous();
    Y = Y.contiguous();
    nUMI = nUMI.contiguous();
    thresh = thresh.contiguous();
    Q_mat = Q_mat.contiguous();
    SQ_mat = SQ_mat.contiguous();
    x_vals = x_vals.contiguous();

    double X_max = x_vals[-1].item<double>();

    const int BLOCK_G = 256;
    dim3 grid(N, (G + BLOCK_G - 1) / BLOCK_G);
    dim3 block(BLOCK_G);

    AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "fused_predict_derivatives", ([&] {
        fused_predict_derivatives_kernel<scalar_t><<<grid, block>>>(
            w.data_ptr<scalar_t>(),
            P.data_ptr<scalar_t>(),
            P_T.data_ptr<scalar_t>(),
            Y.data_ptr<scalar_t>(),
            nUMI.data_ptr<scalar_t>(),
            thresh.data_ptr<scalar_t>(),
            Q_mat.data_ptr<scalar_t>(),
            SQ_mat.data_ptr<scalar_t>(),
            x_vals.data_ptr<scalar_t>(),
            grad.data_ptr<scalar_t>(),
            hess.data_ptr<scalar_t>(),
            N, G, K, N_X, K_val,
            static_cast<scalar_t>(X_max)
        );
    }));

    return {grad, hess.reshape({N, K, K})};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_predict_and_derivatives", &fused_predict_and_derivatives_cuda,
          "Fused prediction + spline interpolation + grad + hess (CUDA)");
}
