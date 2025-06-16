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

    // Get number of rows
    auto count = args.size();

    // Get the parameters (function allow both orders (list, scalar) <=> (scalar, list))
    duckdb::Vector &vector = args.data[0].GetType().id() == LogicalTypeId::LIST ? args.data[0] : args.data[1];
    duckdb::Vector &scalar = args.data[0].GetType().id() == LogicalTypeId::LIST ? args.data[1] : args.data[0];

    // For some scalar operation important to now the order of values (subtract, divide)
    bool first_scalar = args.data[0].GetType().id() == LogicalTypeId::LIST ? false : true;

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

    // Get the actual data as pointer to the first element
    auto data = FlatVector::GetData<TYPE>(*child);
    
    // Stores at the end the overall size of the resulting vector
    auto current_size = ListVector::GetListSize(result);
    // Stores the vector type
    auto vector_type = vector.GetVectorType();
    // Start index of list_entry_t objects (jump to the start entry for a specific row)
    idx_t start_idx = 0;
    // Start index of data (jump to the first value that corresponds to a specific row in input and result)
    idx_t offset = 0;
    idx_t result_offset = 0;
    
    // Function that will be executed for each row
    BinaryExecutor::ExecuteWithNulls<list_entry_t, TYPE, list_entry_t>(
        vector, scalar, result, count,
        [&](const list_entry_t &list, TYPE scalar, ValidityMask &mask, idx_t row_idx) {
            // Reserve space for the result vector
            idx_t new_size = current_size + list.length;
            ListVector::Reserve(result, new_size);

            // Pointer to the input vector (and later to its possible childs)  
            auto *current_vec = &ListVector::GetEntry(vector);
            // Pointer to a vector which should be extended
            auto *result_vec = &result;
            // Number of list_entry_t objects in current dimension
            auto size = list.length;
            // Number of list elements
            idx_t number_elements = list.length;
            // Iterate over each dimension
            while(current_vec->GetType().id() == LogicalTypeId::LIST) {
                // Get the list_entry_t list of current dimension
                auto *metadata = ListVector::GetData(*current_vec);
                idx_t number = 0;
                auto start = vector_type == VectorType::CONSTANT_VECTOR ? 0 : start_idx;
                // Count the number of list elements
                for(idx_t i = start; i < size + start; i++) {
                    number += metadata[i].length;
                }
                // Create a new vector which should be appended to the upper dimension
                Vector child(current_vec->GetType());
                ListVector::Reserve(child, number);
                ListVector::SetListSize(child, number);
                auto *child_metadata = ListVector::GetData(child);
                idx_t data_offset = 0;
                // Adjust list_entry_t list of newly created child
                for(idx_t i = 0; i < size; i++) {
                    child_metadata[i].offset = data_offset;
                    child_metadata[i].length = metadata[i + start].length;
                    data_offset += metadata[i + start].length;
                }
                // Append it to the vector representing upper dimension and go further to next lower dimension
                ListVector::Append(*result_vec, child, size);
                current_vec = &ListVector::GetEntry(*current_vec);
                result_vec = &ListVector::GetEntry(*result_vec);
                size = metadata[0].length;
                number_elements = number;
            }

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
            OP::Operation(data + offset, &scalar, result_data + result_offset, number_elements, true, first_scalar);
            start_idx += list.length;
            result_offset += number_elements;
            if (vector_type != VectorType::CONSTANT_VECTOR) {
                offset += number_elements;
            }
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

    // Get number of rows
    auto count = args.size();

    // Get list vectors (parameters)
    auto &lhs_vec = args.data[0];
    auto &rhs_vec = args.data[1];

    // Get size of list vectors and their content
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

    // Get the actual data as pointer to the first element
    auto lhs_data = FlatVector::GetData<TYPE>(*lhs_child);
    auto rhs_data = FlatVector::GetData<TYPE>(*rhs_child);
    
    // Stores at the end the overall size of the resulting vector
    auto current_size = ListVector::GetListSize(result);
    // Stores the vector type
    auto left_type = lhs_vec.GetVectorType();
    auto right_type = rhs_vec.GetVectorType();
    // Start index of list_entry_t objects (jump to the start entry for a specific row)
    idx_t start_idx = 0;
    // Start index of data (jump to the first value that corresponds to a specific row in both inputs and result)
    idx_t lhs_offset = 0;
    idx_t rhs_offset = 0;
    idx_t result_offset = 0;
    
    // Function that will be executed for each row
    BinaryExecutor::ExecuteWithNulls<list_entry_t, list_entry_t, list_entry_t>(
        lhs_vec, rhs_vec, result, count,
        [&](const list_entry_t &left, const list_entry_t &right, ValidityMask &mask, idx_t row_idx) {
            // Check if the dimensions are equal
            if (left.length != right.length) {
                throw InvalidInputException(
                    "%s: List dimensions must be equal, got left length '%d' and right length '%d'", func_name,
                    left.length, right.length);
            }
            // Reserve space for the result vector
            idx_t new_size = current_size + left.length;
            ListVector::Reserve(result, new_size);

            // Pointer to the input vectors (and later to its possible childs)  
            auto *current_lhs_vec = &ListVector::GetEntry(lhs_vec);
            auto *current_rhs_vec = &ListVector::GetEntry(rhs_vec);
            // Pointer to a vector which should be extended
            auto *result_vec = &result;
            // Number of list_entry_t objects in current dimension
            auto size = left.length;
            // Number of list elements
            idx_t number_elements = left.length;
            // Iterate over each dimension (IMPORTANT: Both parameters must have the same structure)
            while(current_lhs_vec->GetType().id() == LogicalTypeId::LIST) {
                // Get the list_entry_t list of current dimension
                auto *lhs_metadata = ListVector::GetData(*current_lhs_vec);
                auto *rhs_metadata = ListVector::GetData(*current_rhs_vec);
                idx_t number = 0;
                auto lhs_start = left_type == VectorType::CONSTANT_VECTOR ? 0 : start_idx;
                auto rhs_start = right_type == VectorType::CONSTANT_VECTOR ? 0 : start_idx;
                // Count the number of list elements and proof if both parameters have the same structure
                for(idx_t i = lhs_start, j = rhs_start; i < size + lhs_start && j < size + rhs_start; i++, j++) {
                    if (lhs_metadata[i].length != rhs_metadata[j].length) {
                        throw InvalidInputException(
                            "%s: List dimensions must be equal, got left length '%d' and right length '%d'", func_name,
                            lhs_metadata[i].length, rhs_metadata[j].length);
                    }
                    number += lhs_metadata[i].length;
                }
                // Create a new vector which should be appended to the upper dimension
                Vector child(current_lhs_vec->GetType());
                ListVector::Reserve(child, number);
                ListVector::SetListSize(child, number);
                auto *child_metadata = ListVector::GetData(child);
                idx_t data_offset = 0;
                // Adjust list_entry_t list of newly created child
                for(idx_t i = 0; i < size; i++) {
                    child_metadata[i].offset = data_offset;
                    child_metadata[i].length = lhs_metadata[i + lhs_start].length;
                    data_offset += lhs_metadata[i + lhs_start].length;
                }
                // Append it to the vector representing upper dimension and go further to next lower dimension
                ListVector::Append(*result_vec, child, size);
                current_lhs_vec = &ListVector::GetEntry(*current_lhs_vec);
                current_rhs_vec = &ListVector::GetEntry(*current_rhs_vec);
                result_vec = &ListVector::GetEntry(*result_vec);
                size = lhs_metadata[0].length;
                number_elements = number;
            }

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
            OP::Operation(lhs_data + lhs_offset, rhs_data + rhs_offset, result_data + result_offset, number_elements);
            // Adjust control variables
            if (left_type != VectorType::CONSTANT_VECTOR) {
                lhs_offset += number_elements;
            }
            if (right_type != VectorType::CONSTANT_VECTOR) {
                rhs_offset += number_elements;
            }
            result_offset += number_elements;
            start_idx += left.length;
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

    // Get list size
    auto lhs_list_size = ListVector::GetListSize(lhs_vec);
    auto rhs_list_size = ListVector::GetListSize(rhs_vec);

    // Get child vectors
    auto *lhs_child = &ListVector::GetEntry(lhs_vec);
    auto *rhs_child = &ListVector::GetEntry(rhs_vec);
    auto *result_child = &ListVector::GetEntry(result);

    // If the current child vectors contain further lists, select their children until reaching last level
    // And extract their list size
    while(lhs_child->GetType().id() == LogicalTypeId::LIST) {
        lhs_list_size = ListVector::GetListSize(*lhs_child);
        lhs_child = &ListVector::GetEntry(*lhs_child);
    }
    while(rhs_child->GetType().id() == LogicalTypeId::LIST) {
        rhs_list_size = ListVector::GetListSize(*rhs_child);
        rhs_child = &ListVector::GetEntry(*rhs_child);
        result_child = &ListVector::GetEntry(*result_child);
    }

    // Decompress the list vectors (with single values) and flatten them
    rhs_child->Flatten(lhs_list_size);
    lhs_child->Flatten(rhs_list_size);

    D_ASSERT(lhs_child->GetVectorType() == VectorType::FLAT_VECTOR);
    D_ASSERT(rhs_child->GetVectorType() == VectorType::FLAT_VECTOR);

    // NULL values are not allowed
    if (!FlatVector::Validity(*lhs_child).CheckAllValid(lhs_list_size)) {
        throw InvalidInputException("%s: left argument can not contain NULL values", func_name);
    }

    if (!FlatVector::Validity(*rhs_child).CheckAllValid(rhs_list_size)) {
        throw InvalidInputException("%s: right argument can not contain NULL values", func_name);
    }

    // Get the actual data as pointer to the first element
    auto lhs_data = FlatVector::GetData<TYPE>(*lhs_child);
    auto rhs_data = FlatVector::GetData<TYPE>(*rhs_child);
    
    // Stores at the end the overall size of the resulting vector
    auto current_size = ListVector::GetListSize(result);
    // Stores the vector type
    auto left_type = lhs_vec.GetVectorType();
    auto right_type = rhs_vec.GetVectorType();
    // Start index of list_entry_t objects (jump to the start entry for a specific row)
    idx_t lhs_start = 0;
    idx_t rhs_start = 0;
    // Start index of data (jump to values that corresponds to a specific row)
    idx_t lhs_offset = 0;
    idx_t rhs_offset = 0;
    idx_t result_offset = 0;
    
    // Function that will be executed for each row
    BinaryExecutor::ExecuteWithNulls<list_entry_t, list_entry_t, list_entry_t>(
        lhs_vec, rhs_vec, result, count,
        [&](const list_entry_t &left, const list_entry_t &right, ValidityMask &mask, idx_t row_idx) {
            // Extract dimension values
            auto rowsA = left.length;
            auto rowsB = right.length;
            uint64_t colsA = 0;
            uint64_t colsB = 0;

            auto &left_child = ListVector::GetEntry(lhs_vec);
            // If left is multi-dimensional get list metadata of each sublist that contains single elements
            if (left_child.GetType().id() == LogicalTypeId::LIST){
                auto *metadata = ListVector::GetData(left_child);
                // If vector is constant ignore adjusting to the corresponding row
                auto start = left_type == VectorType::CONSTANT_VECTOR ? 0 : lhs_start;
                auto condition = left_type == VectorType::CONSTANT_VECTOR ? left.length : lhs_start + left.length;
                for(idx_t i = start; i < condition; i++) {
                    // Get the size specification and proof if it match with all lists on the same level
                    if (colsA == 0) {
                        colsA = metadata[i].length;
                    }
                    if (colsA != metadata[i].length) {
                        throw InvalidInputException("Left list has an unevenly distributed number of elements");
                    }
                }
            } else {
                colsA = 1;
            }

            auto &right_child = ListVector::GetEntry(rhs_vec);
            // If right is multi-dimensional get list metadata of each sublist that contains single elements
            if (right_child.GetType().id() == LogicalTypeId::LIST) {
                auto *metadata = ListVector::GetData(right_child);
                // If vector is constant ignore adjusting to the corresponding row
                auto start = right_type == VectorType::CONSTANT_VECTOR ? 0 : rhs_start;
                auto condition = right_type == VectorType::CONSTANT_VECTOR ? right.length : rhs_start + right.length;
                for(idx_t i = start; i < condition; i++) {
                    // Get the size specification and proof if it match with all lists on the same level
                    if (colsB == 0) {
                        colsB = metadata[i].length;
                    }
                    if (colsB != metadata[i].length) {
                        throw InvalidInputException("Right list has an unevenly distributed number of elements");
                    }
                }
            } else {
                colsB = 1;
            }
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
            // Set list metadata (of this row)
            list_entry_t result_metadata;
            result_metadata.offset = current_size;
            result_metadata.length = rowsC;

            // If result is two dimensional append sublist
            if (colsC > 1) {
                Vector subvec(duckdb::LogicalType::LIST(rhs_child->GetType()));
                ListVector::Reserve(subvec, rowsC * colsC);
                ListVector::SetListSize(subvec, rowsC * colsC);
                auto* list_data = ListVector::GetData(subvec);
                for (idx_t i = 0; i < rowsC; i++) {
                    list_data[i].offset = i * colsC;
                    list_data[i].length = colsC;
                }
                ListVector::Append(result, subvec, rowsC);
            }
            // Get shared pointer to actual data
            auto result_data = FlatVector::GetData<TYPE>(*result_child);
            
            // If the parameter vectors are empty, set the result vector to NULL
            if (!MatrixMultiplicationOperator::ALLOW_EMPTY && left.length == 0) {
                mask.SetInvalid(row_idx);
                return result_metadata;
            }

            // Perform the actual addition operation
            MatrixMultiplicationOperator::Operation(
                lhs_data + lhs_offset, 
                rhs_data + rhs_offset,
                result_data + result_offset,
                rowsA,
                rowsB,
                colsB
            );
            // Adjust control variable
            current_size += result_metadata.length; 
            lhs_start += left.length;
            rhs_start += right.length;
            if (left_type != VectorType::CONSTANT_VECTOR) {
                lhs_offset += rowsA * colsA;
            }
            if (right_type != VectorType::CONSTANT_VECTOR) {
                rhs_offset += rowsB * colsB;
            }
            result_offset += rowsC * colsC;
            return result_metadata;
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
        set.AddFunction(ScalarFunction({type, list_single}, list_single, ListGenericArithScalar<float, OP>));
        set.AddFunction(ScalarFunction({type, list_double}, list_double, ListGenericArithScalar<float, OP>));
	} else if (type.id() == LogicalTypeId::BFLOAT) {
		set.AddFunction(ScalarFunction({list_single, list_single}, list_single, ListGenericArithList<std::bfloat16_t, OP>));
        set.AddFunction(ScalarFunction({list_double, list_double}, list_double, ListGenericArithList<std::bfloat16_t, OP>));
        set.AddFunction(ScalarFunction({list_single, type}, list_single, ListGenericArithScalar<std::bfloat16_t, OP>));
        set.AddFunction(ScalarFunction({list_double, type}, list_double, ListGenericArithScalar<std::bfloat16_t, OP>));
        set.AddFunction(ScalarFunction({type, list_single}, list_single, ListGenericArithScalar<std::bfloat16_t, OP>));
        set.AddFunction(ScalarFunction({type, list_double}, list_double, ListGenericArithScalar<std::bfloat16_t, OP>));
	} else if (type.id() == LogicalTypeId::DOUBLE) {
		set.AddFunction(ScalarFunction({list_single, list_single}, list_single, ListGenericArithList<double, OP>));
        set.AddFunction(ScalarFunction({list_double, list_double}, list_double, ListGenericArithList<double, OP>));
        set.AddFunction(ScalarFunction({list_single, type}, list_single, ListGenericArithScalar<double, OP>));
        set.AddFunction(ScalarFunction({list_double, type}, list_double, ListGenericArithScalar<double, OP>));
        set.AddFunction(ScalarFunction({type, list_single}, list_single, ListGenericArithScalar<double, OP>));
        set.AddFunction(ScalarFunction({type, list_double}, list_double, ListGenericArithScalar<double, OP>));
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