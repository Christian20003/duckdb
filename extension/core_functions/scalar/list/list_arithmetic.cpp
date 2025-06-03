#include "core_functions/scalar/list_functions.hpp"
#include "core_functions/array_kernels.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

#include <stdfloat>

namespace duckdb {

template <>
void MatrixMultiplicationOperator::Operation<std::bfloat16_t>(const std::bfloat16_t *lhs_data, const std::bfloat16_t *rhs_data, std::bfloat16_t *result_data, const idx_t rowsA, const idx_t rowsB, const idx_t columnsB) {
	idx_t sizeA = rowsA * rowsB;
	idx_t sizeB = rowsB * columnsB;
	idx_t sizeC = rowsA * columnsB;
	std::vector<uint16_t> matrixA;
	std::vector<uint16_t> matrixB;
	std::vector<float> result;
	result.reserve(sizeC);

	for(idx_t i = 0; i < sizeA; i++) {
		matrixA.insert(matrixA.end(), std::bit_cast<uint16_t>(*lhs_data++));
	}
	for(idx_t i = 0; i < sizeB; i++) {
		matrixB.insert(matrixB.end(), std::bit_cast<uint16_t>(*rhs_data++));
	}

	Gemm<uint16_t, float>(rowsA, columnsB, rowsB, matrixA.data(), matrixB.data(), result.data());

	for (idx_t i = 0; i < sizeC; i++) {
		*result_data++ = static_cast<std::bfloat16_t>(result[i]);
	}
};

template <class TYPE, class OP>
static void ListGenericArithScalar(DataChunk &args, ExpressionState &state, Vector &result) {
    // Extract function name
    const auto &lstate = state.Cast<ExecuteFunctionState>();
    const auto &expr = lstate.expr.Cast<BoundFunctionExpression>();
    const auto &func_name = expr.function.name;

    // Get number of parameters
    auto count = args.size();

    // Get the parameters
    duckdb::Vector &vector = args.data[0];
    duckdb::Vector &scalar = args.data[1];

    // Get size of the list vector and its content
    duckdb::idx_t size = ListVector::GetListSize(vector);
    duckdb::Vector *child = &ListVector::GetEntry(vector);
    auto *result_child = &ListVector::GetEntry(result);

    // If the list vector contain nested list vectors, select their children until reaching last level
    while(child->GetType().id() == LogicalTypeId::LIST) {
        size = ListVector::GetListSize(*child);
        child = &ListVector::GetEntry(*child);
        result_child = &ListVector::GetEntry(*result_child);
    }
    
    // Decompress the list vector (with single values) and flatten them
    child->Flatten(size);

    D_ASSERT(child->GetVectorType() == VectorType::FLAT_VECTOR);

    // NULL values are not allowed
    if (!FlatVector::Validity(*child).CheckAllValid(size)) {
        throw InvalidInputException("%s: left argument can not contain NULL values", func_name);
    }

    // Get the actual data as shared pointer to the first element
    auto data = FlatVector::GetData<TYPE>(*child);
    
    auto current_size = ListVector::GetListSize(result);
    
    // Function that will be executed for each row
    BinaryExecutor::ExecuteWithNulls<list_entry_t, TYPE, list_entry_t>(
        vector, scalar, result, count,
        [&](const list_entry_t &list, TYPE scalar, ValidityMask &mask, idx_t row_idx) {
            // Reserve space for the result vector
            idx_t new_size = current_size + list.length;
            ListVector::Reserve(result, new_size);
            // TODO: Maybe find better solution than copy
            // Is currently needed, to ensure that sublists have a valid offset
            VectorOperations::Copy(ListVector::GetEntry(vector), ListVector::GetEntry(result), list.offset + list.length, list.offset, current_size);
            auto result_data = FlatVector::GetData<TYPE>(*result_child);
            
            // Specify metadata for the result vector
            list_entry_t result_list;
            result_list.offset = current_size;
            result_list.length = list.length;
            current_size += list.length;
            
            // If the parameter vectors are empty, set the result vector to NULL
            if (!OP::ALLOW_EMPTY && list.length == 0) {
                mask.SetInvalid(row_idx);
                return result_list;
            }

            // Perform the actual addition operation 
            OP::Operation(data + list.offset, &scalar, result_data + result_list.offset, size, true);
            return result_list;
        });

    if (args.AllConstant()) {
        result.SetVectorType(VectorType::CONSTANT_VECTOR);
    }
    ListVector::SetListSize(result, current_size);
}

template <class TYPE, class OP>
static void ListGenericArithList(DataChunk &args, ExpressionState &state, Vector &result) {
    // Extract function name
    const auto &lstate = state.Cast<ExecuteFunctionState>();
    const auto &expr = lstate.expr.Cast<BoundFunctionExpression>();
    const auto &func_name = expr.function.name;

    // Get number of parameters
    auto count = args.size();

    // Get the list vectors (parameters)
    auto &lhs_vec = args.data[0];
    auto &rhs_vec = args.data[1];

    // Get size of the list vectors and their content
    auto lhs_count = ListVector::GetListSize(lhs_vec);
    auto rhs_count = ListVector::GetListSize(rhs_vec);
    auto *lhs_child = &ListVector::GetEntry(lhs_vec);
    auto *rhs_child = &ListVector::GetEntry(rhs_vec);
    auto *result_child = &ListVector::GetEntry(result);

    // If the list vectors contain nested list vectors, select their children until reaching last level
    while(lhs_child->GetType().id() == LogicalTypeId::LIST) {
        lhs_count = ListVector::GetListSize(*lhs_child);
        rhs_count = ListVector::GetListSize(*rhs_child);
        lhs_child = &ListVector::GetEntry(*lhs_child);
        rhs_child = &ListVector::GetEntry(*rhs_child);
        result_child = &ListVector::GetEntry(*result_child);
    }
    // Decompress the list vectors (with single values) and flatten them
    rhs_child->Flatten(rhs_count);
    lhs_child->Flatten(lhs_count);

    D_ASSERT(lhs_child->GetVectorType() == VectorType::FLAT_VECTOR);
    D_ASSERT(rhs_child->GetVectorType() == VectorType::FLAT_VECTOR);

    // NULL values are not allowed
    if (!FlatVector::Validity(*lhs_child).CheckAllValid(lhs_count)) {
        throw InvalidInputException("%s: left argument can not contain NULL values", func_name);
    }

    if (!FlatVector::Validity(*rhs_child).CheckAllValid(rhs_count)) {
        throw InvalidInputException("%s: right argument can not contain NULL values", func_name);
    }

    // Get the actual data as shared pointer to the first element
    auto lhs_data = FlatVector::GetData<TYPE>(*lhs_child);
    auto rhs_data = FlatVector::GetData<TYPE>(*rhs_child);
    
    auto current_size = ListVector::GetListSize(result);
    
    // Function that will be executed for each row
    BinaryExecutor::ExecuteWithNulls<list_entry_t, list_entry_t, list_entry_t>(
        lhs_vec, rhs_vec, result, count,
        [&](const list_entry_t &left, const list_entry_t &right, ValidityMask &mask, idx_t row_idx) {
            // Check if the dimensions are equal
            if (left.length != right.length) {
                throw InvalidInputException(
                    "%s: first list dimensions must be equal, got left length '%d' and right length '%d'", func_name,
                    left.length, right.length);
                }
            if (lhs_count != rhs_count) {
                throw InvalidInputException(
                    "%s: last list dimensions must be equal, got left length '%d' and right length '%d'", func_name,
                    lhs_count, rhs_count);
            }
            // Reserve space for the result vector
            idx_t new_size = current_size + left.length;
            ListVector::Reserve(result, new_size);
            // TODO: Maybe find better solution than copy
            // Is currently needed, to ensure that sublists have a valid offset
            VectorOperations::Copy(ListVector::GetEntry(lhs_vec), ListVector::GetEntry(result), left.offset + left.length, left.offset, current_size);
            auto result_data = FlatVector::GetData<TYPE>(*result_child);
            
            // Specify metadata for the result vector
            list_entry_t result_list;
            result_list.offset = current_size;
            result_list.length = left.length;
            current_size += left.length;
            
            // If the parameter vectors are empty, set the result vector to NULL
            if (!OP::ALLOW_EMPTY && left.length == 0) {
                mask.SetInvalid(row_idx);
                return result_list;
            }

            // Perform the actual addition operation 
            OP::Operation(lhs_data + left.offset, rhs_data + right.offset, result_data + result_list.offset, lhs_count);
            return result_list;
        });

    if (args.AllConstant()) {
        result.SetVectorType(VectorType::CONSTANT_VECTOR);
    }
    ListVector::SetListSize(result, current_size);
}

template <class TYPE>
static void ListMatrixMul(DataChunk &args, ExpressionState &state, Vector &result) {
    // Extract function name
    const auto &lstate = state.Cast<ExecuteFunctionState>();
    const auto &expr = lstate.expr.Cast<BoundFunctionExpression>();
    const auto &func_name = expr.function.name;

    // Get number of rows
    auto count = args.size();

    // Get function parameters (IMPORTANT: This will include all rows from a chunk)
    auto &lhs_vec = args.data[0];
    auto &rhs_vec = args.data[1];

    // Get list size of first dimension
    auto left_outer_size = ListVector::GetListSize(lhs_vec);
    auto right_outer_size = ListVector::GetListSize(rhs_vec);
    // Will store the list size of second dimension
    duckdb::idx_t left_inner_size = 0;
    duckdb::idx_t right_inner_size = 0;

    // Get child vectors
    auto *lhs_child = &ListVector::GetEntry(lhs_vec);
    auto *rhs_child = &ListVector::GetEntry(rhs_vec);
    auto *result_child = &ListVector::GetEntry(result);

    // If the current child vectors contain further lists, select their children until reaching last level
    // And extract their list size
    while(lhs_child->GetType().id() == LogicalTypeId::LIST) {
        left_inner_size = ListVector::GetListSize(*lhs_child);
        lhs_child = &ListVector::GetEntry(*lhs_child);
    }
    while(rhs_child->GetType().id() == LogicalTypeId::LIST) {
        right_inner_size = ListVector::GetListSize(*rhs_child);
        rhs_child = &ListVector::GetEntry(*rhs_child);
        result_child = &ListVector::GetEntry(*result_child);
    }

    // Decompress the list vectors (with single values) and flatten them
    auto l_size = left_inner_size == 0 ? left_outer_size : left_inner_size;
    auto r_size = right_inner_size == 0 ? right_outer_size : right_inner_size;
    rhs_child->Flatten(l_size);
    lhs_child->Flatten(r_size);

    D_ASSERT(lhs_child->GetVectorType() == VectorType::FLAT_VECTOR);
    D_ASSERT(rhs_child->GetVectorType() == VectorType::FLAT_VECTOR);

    // NULL values are not allowed
    if (!FlatVector::Validity(*lhs_child).CheckAllValid(l_size)) {
        throw InvalidInputException("%s: left argument can not contain NULL values", func_name);
    }

    if (!FlatVector::Validity(*rhs_child).CheckAllValid(r_size)) {
        throw InvalidInputException("%s: right argument can not contain NULL values", func_name);
    }

    // Reset second dimension value if list has only one dimension
    left_inner_size = left_inner_size == 0 ? 1 : left_inner_size;
    right_inner_size = right_inner_size == 0 ? 1 : right_inner_size;

    // Get the actual data as shared pointer to the first element
    auto lhs_data = FlatVector::GetData<TYPE>(*lhs_child);
    auto rhs_data = FlatVector::GetData<TYPE>(*rhs_child);
    
    // Create control variable 
    auto current_size = ListVector::GetListSize(result);
    
    // Function that will be executed for each row
    BinaryExecutor::ExecuteWithNulls<list_entry_t, list_entry_t, list_entry_t>(
        lhs_vec, rhs_vec, result, count,
        [&](const list_entry_t &left, const list_entry_t &right, ValidityMask &mask, idx_t row_idx) {
            // Extract dimension values
            auto rowsA = left.length;
            auto colsA = left_inner_size != 1 ? left_inner_size / left_outer_size : left_inner_size;
            auto rowsB = right.length;
            auto colsB = right_inner_size != 1 ? right_inner_size / right_outer_size : right_inner_size;
            auto rowsC = rowsA;
            auto colsC = colsB;
            // Check if the dimensions are valid for matrix multiplication
            if (colsA != rowsB) {
                throw InvalidInputException(
                    "%s: invalid dimension structure for matrix multiplication, got '%d'x'%d' and '%d'x'%d'", func_name,
                    rowsA, colsA, rowsB, colsB);
            }

            // Reserve space for the result vector
            idx_t new_size = current_size + rowsC;
            ListVector::Reserve(result, new_size);
            // Set list metadata
            auto result_metadata = ListVector::GetData(result);
            result_metadata->offset = current_size;
            result_metadata->length = rowsC;

            // If result is two dimensional append sublists
            if (colsC > 1) {
                for (idx_t i = 0; i < rowsC; i++) {
                    Vector subvec(duckdb::LogicalType::LIST(rhs_child->GetType()));
                    ListVector::Reserve(subvec, colsC);
                    ListVector::SetListSize(subvec, colsC);
                    auto* list_data = ListVector::GetData(subvec);
                    list_data->offset = 0;
                    list_data->length = colsC;
                    ListVector::Append(result, subvec, 1);
                }
                // TODO: Not sure if needed 
                /* auto final_vec_data = FlatVector::GetData<list_entry_t>(ListVector::GetEntry(result));
                for (idx_t i = 0; i < rowsC; i++) {
                    final_vec_data[i].offset = i * colsC;
                    final_vec_data[i].length = colsC;
                } */
            }
            // Get shared pointer to actual data
            auto result_data = FlatVector::GetData<TYPE>(*result_child);
            
            // If the parameter vectors are empty, set the result vector to NULL
            if (!MatrixMultiplicationOperator::ALLOW_EMPTY && left.length == 0) {
                mask.SetInvalid(row_idx);
                return *result_metadata;
            }

            // Perform the actual addition operation
            MatrixMultiplicationOperator::Operation(
                lhs_data + current_size * colsA, 
                rhs_data + current_size * colsB,
                result_data + current_size * colsC,
                rowsA,
                rowsB,
                colsB
            );
            // Adjust control variable
            current_size += result_metadata->length; 
            return *result_metadata;
        });

    if (args.AllConstant()) {
        result.SetVectorType(VectorType::CONSTANT_VECTOR);
    }
    ListVector::SetListSize(result, current_size);
}

//-------------------------------------------------------------------------
// Function Registration
//-------------------------------------------------------------------------

template <class OP>
static void AddListArithFunction(ScalarFunctionSet &set, const LogicalType &type) {
	const auto list_single = LogicalType::LIST(type);
    const auto list_double = LogicalType::LIST(LogicalType::LIST(type));
	if (type.id() == LogicalTypeId::FLOAT) {
		set.AddFunction(ScalarFunction({list_single, list_single}, list_single, ListGenericArithList<float, OP>));
        set.AddFunction(ScalarFunction({list_double, list_double}, list_double, ListGenericArithList<float, OP>));
        set.AddFunction(ScalarFunction({list_single, type}, list_single, ListGenericArithScalar<float, OP>));
        set.AddFunction(ScalarFunction({list_double, type}, list_double, ListGenericArithScalar<float, OP>));
	} else if (type.id() == LogicalTypeId::BFLOAT) {
		set.AddFunction(ScalarFunction({list_single, list_single}, list_single, ListGenericArithList<std::bfloat16_t, OP>));
        set.AddFunction(ScalarFunction({list_double, list_double}, list_double, ListGenericArithList<std::bfloat16_t, OP>));
        set.AddFunction(ScalarFunction({list_single, type}, list_single, ListGenericArithScalar<std::bfloat16_t, OP>));
        set.AddFunction(ScalarFunction({list_double, type}, list_double, ListGenericArithScalar<std::bfloat16_t, OP>));
	} else if (type.id() == LogicalTypeId::DOUBLE) {
		set.AddFunction(ScalarFunction({list_single, list_single}, list_single, ListGenericArithList<double, OP>));
        set.AddFunction(ScalarFunction({list_double, list_double}, list_double, ListGenericArithList<double, OP>));
        set.AddFunction(ScalarFunction({list_single, type}, list_single, ListGenericArithScalar<double, OP>));
        set.AddFunction(ScalarFunction({list_double, type}, list_double, ListGenericArithScalar<double, OP>));
	} else {
		throw NotImplementedException("List function not implemented for type %s", type.ToString());
	}
}

ScalarFunctionSet ListArithAddFun::GetFunctions() {
	ScalarFunctionSet set("list_add");
	for (auto &type : LogicalType::Real()) {
		AddListArithFunction<AddOperator>(set, type);
	}
	for (auto &func : set.functions) {
		BaseScalarFunction::SetReturnsError(func);
	}
	return set;
}

ScalarFunctionSet ListArithSubFun::GetFunctions() {
	ScalarFunctionSet set("list_sub");
	for (auto &type : LogicalType::Real()) {
		AddListArithFunction<SubOperator>(set, type);
	}
	for (auto &func : set.functions) {
		BaseScalarFunction::SetReturnsError(func);
	}
	return set;
}

ScalarFunctionSet ListArithMulFun::GetFunctions() {
	ScalarFunctionSet set("list_mul");
	for (auto &type : LogicalType::Real()) {
		AddListArithFunction<MulOperator>(set, type);
	}
	for (auto &func : set.functions) {
		BaseScalarFunction::SetReturnsError(func);
	}
	return set;
}

ScalarFunctionSet ListArithDivFun::GetFunctions() {
	ScalarFunctionSet set("list_div");
	for (auto &type : LogicalType::Real()) {
		AddListArithFunction<DivOperator>(set, type);
	}
	for (auto &func : set.functions) {
		BaseScalarFunction::SetReturnsError(func);
	}
	return set;
}

ScalarFunctionSet ListArithMMulFun::GetFunctions() {
	ScalarFunctionSet set("list_mmul");
	for (auto &type : LogicalType::Real()) {
        const auto list_single = LogicalType::LIST(type);
        const auto list_double = LogicalType::LIST(LogicalType::LIST(type));
        const auto metadataType = LogicalType::INTEGER;
        if (type.id() == LogicalTypeId::FLOAT) {
            set.AddFunction(ScalarFunction({list_double, list_double}, list_double, ListMatrixMul<float>));
            set.AddFunction(ScalarFunction({list_double, list_single}, list_single, ListMatrixMul<float>));
            set.AddFunction(ScalarFunction({list_single, list_double}, list_double, ListMatrixMul<float>));
        } else if (type.id() == LogicalTypeId::BFLOAT) {
            set.AddFunction(ScalarFunction({list_double, list_double}, list_double, ListMatrixMul<std::bfloat16_t>));
            set.AddFunction(ScalarFunction({list_double, list_single}, list_single, ListMatrixMul<std::bfloat16_t>));
            set.AddFunction(ScalarFunction({list_single, list_double}, list_double, ListMatrixMul<std::bfloat16_t>));
        } else if (type.id() == LogicalTypeId::DOUBLE) {
            set.AddFunction(ScalarFunction({list_double, list_double}, list_double, ListMatrixMul<double>));
            set.AddFunction(ScalarFunction({list_double, list_single}, list_single, ListMatrixMul<double>));
            set.AddFunction(ScalarFunction({list_single, list_double}, list_double, ListMatrixMul<double>));
        }
	}
	for (auto &func : set.functions) {
		BaseScalarFunction::SetReturnsError(func);
	}
	return set;
}
}