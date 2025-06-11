#pragma once
#include "duckdb/common/typedefs.hpp"
#include "duckdb/common/algorithm.hpp"
#include <cmath>
#include "cblas.h"

namespace duckdb {
//-------------------------------------------------------------------------
// Activation Functions
//-------------------------------------------------------------------------

struct SigmoidOperator {
	static constexpr bool ALLOW_EMPTY = true;

	template <class TYPE>
	static void Operation(const TYPE *data, TYPE *result, const idx_t count) {
		for(idx_t i = 0; i < count; i++) {
			*result++ = 1 / (1 + std::exp(-(*data++)));
		}
	}
};

//-------------------------------------------------------------------------
// Modification Operation
//-------------------------------------------------------------------------

struct TransposeOperator {
	static constexpr bool ALLOW_EMPTY = true;

	template <class TYPE>
	static void Operation(const TYPE *data, TYPE *result, const idx_t rows, const idx_t cols) {
		for(idx_t column = 0; column < cols; column++) {
			for(idx_t row = 0; row < rows*cols; row+=cols) {
				*result++ = *(data + column + row);
			}
		}
	}
};

//-------------------------------------------------------------------------
// Arithmetic Operations
//-------------------------------------------------------------------------

template <class TYPE, class RETURN_TYPE>
inline void Gemm(int rowsA, int columnsB, int rowsB, const TYPE *A, const TYPE *B, RETURN_TYPE *C) {
    static_assert(sizeof(TYPE) == 0, "Gemm not implemented for this type");
}

template <>
inline void Gemm<double, double>(int rowsA, int columnsB, int rowsB, const double *A, const double *B, double *C) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rowsA, columnsB, rowsB, 1.0, A, rowsB, B, columnsB, 0.0, C, columnsB);
}

template <>
inline void Gemm<float, float>(int rowsA, int columnsB, int rowsB, const float *A, const float *B, float *C) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rowsA, columnsB, rowsB, 1.0f, A, rowsB, B, columnsB, 0.0f, C, columnsB);
}

template <>
inline void Gemm<uint16_t, float>(int rowsA, int columnsB, int rowsB, const uint16_t *A, const uint16_t *B, float *C) {
    cblas_sbgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rowsA, columnsB, rowsB, 1.0f, A, rowsB, B, columnsB, 0.0f, C, columnsB);
}

struct MatrixMultiplicationOperator {
	static constexpr bool ALLOW_EMPTY = false;

	template <class TYPE>
	static void Operation(const TYPE *lhs_data, const TYPE *rhs_data, TYPE *result_data, const idx_t rowsA, const idx_t rowsB, const idx_t columnsB) {
		idx_t sizeC = rowsA * columnsB;
		std::vector<TYPE> result;
		result.reserve(sizeC);

		Gemm<TYPE, TYPE>(rowsA, columnsB, rowsB, lhs_data, rhs_data, result.data());

		for (idx_t i = 0; i < sizeC; i++) {
			*result_data++ = result[i];
		}
	}
};

struct AddOperator {
	static constexpr bool ALLOW_EMPTY = true;

	template <class TYPE>
	static void Operation(const TYPE *lhs_data, const TYPE *rhs_data, TYPE *result_data, const idx_t count, const bool withScalar = false, const bool scalarFirst = false) {
		auto lhs_ptr = lhs_data;
		auto rhs_ptr = rhs_data;
		auto result_ptr = result_data;

		for (idx_t i = 0; i < count; i++) {
			const TYPE x = *lhs_ptr++;
			const TYPE y = withScalar ? *rhs_ptr : *rhs_ptr++;
			if (withScalar && scalarFirst) {
				*result_ptr++ = y + x;
			} else {
				*result_ptr++ = x + y;
			}
		}
	}
};

struct SubOperator {
	static constexpr bool ALLOW_EMPTY = true;

	template <class TYPE>
	static void Operation(const TYPE *lhs_data, const TYPE *rhs_data, TYPE *result_data, const idx_t count, const bool withScalar = false, const bool scalarFirst = false) {
		auto lhs_ptr = lhs_data;
		auto rhs_ptr = rhs_data;
		auto result_ptr = result_data;

		for (idx_t i = 0; i < count; i++) {
			const TYPE x = *lhs_ptr++;
			const TYPE y = withScalar ? *rhs_ptr : *rhs_ptr++;
			if (withScalar && scalarFirst) {
				*result_ptr++ = y - x;
			} else {
				*result_ptr++ = x - y;
			}
		}
	}
};

struct MulOperator {
	static constexpr bool ALLOW_EMPTY = true;

	template <class TYPE>
	static void Operation(const TYPE *lhs_data, const TYPE *rhs_data, TYPE *result_data, const idx_t count, const bool withScalar = false, const bool scalarFirst = false) {
		auto lhs_ptr = lhs_data;
		auto rhs_ptr = rhs_data;
		auto result_ptr = result_data;

		for (idx_t i = 0; i < count; i++) {
			const TYPE x = *lhs_ptr++;
			const TYPE y = withScalar ? *rhs_ptr : *rhs_ptr++;
			if (withScalar && scalarFirst) {
				*result_ptr++ = y * x;
			} else {
				*result_ptr++ = x * y;
			}
		}
	}
};

struct DivOperator {
	static constexpr bool ALLOW_EMPTY = true;

	template <class TYPE>
	static void Operation(const TYPE *lhs_data, const TYPE *rhs_data, TYPE *result_data, const idx_t count, const bool withScalar = false, const bool scalarFirst = false) {
		auto lhs_ptr = lhs_data;
		auto rhs_ptr = rhs_data;
		auto result_ptr = result_data;

		for (idx_t i = 0; i < count; i++) {
			const TYPE x = *lhs_ptr++;
			const TYPE y = withScalar ? *rhs_ptr : *rhs_ptr++;
			if (withScalar && scalarFirst) {
				*result_ptr++ = y / x;
			} else {
				*result_ptr++ = x / y;
			}
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
