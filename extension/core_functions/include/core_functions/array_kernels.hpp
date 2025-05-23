#pragma once
#include "duckdb/common/typedefs.hpp"
#include "duckdb/common/algorithm.hpp"
#include <cmath>
#include "cblas.h"
#include <iostream>

namespace duckdb {
//-------------------------------------------------------------------------
// Arithmetic Operations
//-------------------------------------------------------------------------

struct AddOperator {
	static constexpr bool ALLOW_EMPTY = true;

	template <class TYPE>
	static void Operation(const TYPE *lhs_data, const TYPE *rhs_data, TYPE *result_data, const idx_t count, const bool withScalar = false) {
		auto lhs_ptr = lhs_data;
		auto rhs_ptr = rhs_data;
		auto result_ptr = result_data;

		for (idx_t i = 0; i < count; i++) {
			const TYPE x = *lhs_ptr++;
			const TYPE y = withScalar ? *rhs_ptr : *rhs_ptr++;
			*result_ptr++ = x + y;
		}

		/* // Define the dimensions of the matrices
		const int M = 2; // Number of rows in A and C
		const int N = 3; // Number of columns in B and C
		const int K = 2; // Number of columns in A and rows in B
	
		// Define matrices A (MxK) and B (KxN)
		double A[M * K] = {
			1.0, 2.0, // First row of A
			3.0, 4.0  // Second row of A
		};
	
		double B[K * N] = {
			5.0, 6.0, 7.0, // First row of B
			8.0, 9.0, 10.0 // Second row of B
		};
	
		// Result matrix C (MxN)
		double C[M * N] = {0.0}; // Initialize C to zero
	
		// Perform matrix multiplication: C = A * B
		// CBLAS function: cblas_dgemm
		// Parameters:
		// CblasRowMajor: Row-major order
		// CblasNoTrans: No transpose for A
		// CblasNoTrans: No transpose for B
		// M: Number of rows in A
		// N: Number of columns in B
		// K: Number of columns in A (or rows in B)
		// alpha: Scalar multiplier for the product
		// A: Pointer to matrix A
		// lda: Leading dimension of A
		// B: Pointer to matrix B
		// ldb: Leading dimension of B
		// beta: Scalar multiplier for C
		// C: Pointer to matrix C
		// ldc: Leading dimension of C
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					M, N, K, 1.0, A, K, B, N, 0.0, C, N);
	
		// Print the result matrix C
		std::cout << "Result of C = A * B:" << std::endl;
		for (int i = 0; i < M; ++i) {
			for (int j = 0; j < N; ++j) {
				std::cout << C[i * N + j] << " ";
			}
			std::cout << std::endl;
		} */
	}
};

struct SubOperator {
	static constexpr bool ALLOW_EMPTY = true;

	template <class TYPE>
	static void Operation(const TYPE *lhs_data, const TYPE *rhs_data, TYPE *result_data, const idx_t count, const bool withScalar = false) {
		auto lhs_ptr = lhs_data;
		auto rhs_ptr = rhs_data;
		auto result_ptr = result_data;

		for (idx_t i = 0; i < count; i++) {
			const TYPE x = *lhs_ptr++;
			const TYPE y = withScalar ? *rhs_ptr : *rhs_ptr++;
			*result_ptr++ = x - y;
		}
	}
};

struct MulOperator {
	static constexpr bool ALLOW_EMPTY = true;

	template <class TYPE>
	static void Operation(const TYPE *lhs_data, const TYPE *rhs_data, TYPE *result_data, const idx_t count, const bool withScalar = false) {
		auto lhs_ptr = lhs_data;
		auto rhs_ptr = rhs_data;
		auto result_ptr = result_data;

		for (idx_t i = 0; i < count; i++) {
			const TYPE x = *lhs_ptr++;
			const TYPE y = withScalar ? *rhs_ptr : *rhs_ptr++;
			*result_ptr++ = x * y;
		}
	}
};

struct DivOperator {
	static constexpr bool ALLOW_EMPTY = true;

	template <class TYPE>
	static void Operation(const TYPE *lhs_data, const TYPE *rhs_data, TYPE *result_data, const idx_t count, const bool withScalar = false) {
		auto lhs_ptr = lhs_data;
		auto rhs_ptr = rhs_data;
		auto result_ptr = result_data;

		for (idx_t i = 0; i < count; i++) {
			const TYPE x = *lhs_ptr++;
			const TYPE y = withScalar ? *rhs_ptr : *rhs_ptr++;
			*result_ptr++ = x / y;
		}
	}
};

//-------------------------------------------------------------------------
// Folding Operations
//-------------------------------------------------------------------------
struct InnerProductOp {
	static constexpr bool ALLOW_EMPTY = true;

	template <class TYPE>
	static TYPE Operation(const TYPE *lhs_data, const TYPE *rhs_data, const idx_t count) {

		TYPE result = 0;

		auto lhs_ptr = lhs_data;
		auto rhs_ptr = rhs_data;

		for (idx_t i = 0; i < count; i++) {
			const auto x = *lhs_ptr++;
			const auto y = *rhs_ptr++;
			result += x * y;
		}

		return result;
	}
};

struct NegativeInnerProductOp {
	static constexpr bool ALLOW_EMPTY = true;

	template <class TYPE>
	static TYPE Operation(const TYPE *lhs_data, const TYPE *rhs_data, const idx_t count) {
		return -InnerProductOp::Operation(lhs_data, rhs_data, count);
	}
};

struct CosineSimilarityOp {
	static constexpr bool ALLOW_EMPTY = false;

	template <class TYPE>
	static TYPE Operation(const TYPE *lhs_data, const TYPE *rhs_data, const idx_t count) {

		TYPE distance = 0;
		TYPE norm_l = 0;
		TYPE norm_r = 0;

		auto l_ptr = lhs_data;
		auto r_ptr = rhs_data;

		for (idx_t i = 0; i < count; i++) {
			const auto x = *l_ptr++;
			const auto y = *r_ptr++;
			distance += x * y;
			norm_l += x * x;
			norm_r += y * y;
		}

		auto similarity = distance / std::sqrt(norm_l * norm_r);
		return std::max(static_cast<TYPE>(-1.0), std::min(similarity, static_cast<TYPE>(1.0)));
	}
};

struct CosineDistanceOp {
	static constexpr bool ALLOW_EMPTY = false;

	template <class TYPE>
	static TYPE Operation(const TYPE *lhs_data, const TYPE *rhs_data, const idx_t count) {
		return static_cast<TYPE>(1.0) - CosineSimilarityOp::Operation(lhs_data, rhs_data, count);
	}
};

struct DistanceSquaredOp {
	static constexpr bool ALLOW_EMPTY = true;

	template <class TYPE>
	static TYPE Operation(const TYPE *lhs_data, const TYPE *rhs_data, const idx_t count) {

		TYPE distance = 0;

		auto l_ptr = lhs_data;
		auto r_ptr = rhs_data;

		for (idx_t i = 0; i < count; i++) {
			const auto x = *l_ptr++;
			const auto y = *r_ptr++;
			const auto diff = x - y;
			distance += diff * diff;
		}

		return distance;
	}
};

struct DistanceOp {
	static constexpr bool ALLOW_EMPTY = true;

	template <class TYPE>
	static TYPE Operation(const TYPE *lhs_data, const TYPE *rhs_data, const idx_t count) {
		return std::sqrt(DistanceSquaredOp::Operation(lhs_data, rhs_data, count));
	}
};

} // namespace duckdb
