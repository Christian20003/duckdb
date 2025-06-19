#include "core_functions/scalar/list_functions.hpp"
#include "core_functions/array_kernels.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

#include <stdfloat>

namespace duckdb {

// Specific implementation for matrix multiplication with bfloat (openBlas)
template <>
void MatrixMultiplicationOperator::Operation<std::bfloat16_t>(
    const std::bfloat16_t *lhs_data, 
    const std::bfloat16_t *rhs_data, 
    std::bfloat16_t *result_data, 
    const idx_t rowsA, 
    const idx_t rowsB, 
    const idx_t columnsB) 
{
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

/**
 * This function executes scalar operations on lists
 */
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

    // Select the child vectors which are not of type LIST (vectors which contain elements of type TYPE)
    duckdb::idx_t size = ListVector::GetListSize(vector);
    duckdb::Vector *child = &ListVector::GetEntry(vector);
    auto *result_child = &ListVector::GetEntry(result);
    while(child->GetType().id() == LogicalTypeId::LIST) {
        size = ListVector::GetListSize(*child);
        child = &ListVector::GetEntry(*child);
        result_child = &ListVector::GetEntry(*result_child);
    }
    
    // Transform ListVector into FlatVector to get access to the elements
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
    // Start index of data (jump to the first value that corresponds to a specific row in result)
    idx_t result_offset = 0;

    // Copy input vector to result, because input structure == output structure
    // Rebuilding result vector from scratch should be less efficient
    VectorOperations::Copy(vector, result, count, 0, 0);
    
    // Function that will be executed for each row
    BinaryExecutor::ExecuteWithNulls<list_entry_t, TYPE, list_entry_t>(
        vector, scalar, result, count,
        [&](const list_entry_t &list, TYPE scalar, ValidityMask &mask, idx_t row_idx) {
            // Reserve space for the result vector
            idx_t new_size = current_size + list.length;
            ListVector::Reserve(result, new_size);

            // Specify metadata for the result vector
            list_entry_t result_list;
            result_list.offset = current_size;
            result_list.length = list.length;

            // If the parameter vectors are empty, set the result to NULL
            if (!OP::ALLOW_EMPTY && list.length == 0) {
                mask.SetInvalid(row_idx);
                return result_list;
            }

            // Pointer to the input vector (and later to its possible childs)  
            auto *current_vec = &ListVector::GetEntry(vector);
            // Value which stores the current offset of a specific dimension
            idx_t offset = list.offset;
            // Value which store the current length of a specific dimension
            idx_t length = list.length;
            // Iterate over each dimension
            while(current_vec->GetType().id() == LogicalTypeId::LIST) {
                // Value which stores the number of elements in the current dimension and row
                idx_t sublist_length = 0;
                // Value which stores the number of elements in the current dimension and previous rows
                idx_t prev_length = 0;
                // Get the list_entry_t list of current dimension
                auto *metadata = ListVector::GetData(*current_vec);
                // Count the number of list elements
                for(idx_t i = 0; i < offset + length; i++) {
                    // Adjust number elements of this row
                    if (i >= offset) {
                        sublist_length += metadata[i].length;
                    // Adjust number elements of previous rows
                    } else {
                        prev_length += metadata[i].length;
                    }
                }
                current_vec = &ListVector::GetEntry(*current_vec);
                // Adjust offset for next dimension
                offset = prev_length;
                // Adjust length for next dimension
                length = sublist_length;
            }
            // Pointer to the first element in result
            auto result_data = FlatVector::GetData<TYPE>(*result_child);

            // Perform the actual scalar operation 
            OP::Operation(data + offset, &scalar, result_data + result_offset, length, true, first_scalar);
            // Adjust result offsets
            result_offset += length;
            current_size += list.length;
            return result_list;
        });

    if (args.AllConstant()) {
        result.SetVectorType(VectorType::CONSTANT_VECTOR);
    }
    ListVector::SetListSize(result, current_size);
}

/**
 * This function executes elementwise operations on lists
 */
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

    // Select the child vectors which are not of type LIST (vectors which contain elements of type TYPE)
    auto lhs_count = ListVector::GetListSize(lhs_vec);
    auto rhs_count = ListVector::GetListSize(rhs_vec);
    auto *lhs_child = &ListVector::GetEntry(lhs_vec);
    auto *rhs_child = &ListVector::GetEntry(rhs_vec);
    auto *result_child = &ListVector::GetEntry(result);
    while(lhs_child->GetType().id() == LogicalTypeId::LIST) {
        lhs_count = ListVector::GetListSize(*lhs_child);
        rhs_count = ListVector::GetListSize(*rhs_child);
        lhs_child = &ListVector::GetEntry(*lhs_child);
        rhs_child = &ListVector::GetEntry(*rhs_child);
        result_child = &ListVector::GetEntry(*result_child);
    }

    // Transform ListVector into FlatVector to get access to the elements
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
    // Start index of data (jump to the first value that corresponds to a specific row in result)
    idx_t result_offset = 0;

    // Copy input vector to result, because input structure == output structure
    // Rebuilding result vector from scratch should be less efficient
    VectorOperations::Copy(lhs_vec, result, count, 0, 0);
    
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

            // Reserve space for the result vector
            idx_t new_size = current_size + left.length;
            ListVector::Reserve(result, new_size);

            // Pointer to the input vectors (and later to its possible childs)  
            auto *current_lhs_vec = &ListVector::GetEntry(lhs_vec);
            auto *current_rhs_vec = &ListVector::GetEntry(rhs_vec);
            // Values which store the current offset of a specific dimension
            idx_t lhs_offset = left.offset;
            idx_t rhs_offset = right.offset;
            // Values which store the current length of a specific dimension
            idx_t lhs_length = left.length;
            idx_t rhs_length = right.length;
            // Iterate over each dimension (IMPORTANT: Both parameters must have the same structure)
            while(current_lhs_vec->GetType().id() == LogicalTypeId::LIST) {
                // Values which store the number of elements in the current dimension and row
                idx_t lhs_sublist_length = 0;
                idx_t rhs_sublist_length = 0;
                // Values which store the number of elements in the current dimension and previous rows
                idx_t lhs_prev_length = 0;
                idx_t rhs_prev_length = 0;
                // Get the list_entry_t list of current dimension
                auto *lhs_metadata = ListVector::GetData(*current_lhs_vec);
                auto *rhs_metadata = ListVector::GetData(*current_rhs_vec);
                // Count the number of list elements and proof if both parameters have the same structure
                for(idx_t i = 0, j = 0; i < lhs_offset + lhs_length && j < rhs_offset + rhs_length; i++, j++) {
                    if (lhs_metadata[i].length != rhs_metadata[j].length && i >= lhs_offset && j >= rhs_offset) {
                        throw InvalidInputException(
                            "%s: List dimensions must be equal, got left length '%d' and right length '%d'", func_name,
                            lhs_metadata[i].length, rhs_metadata[j].length);
                    }
                    // Adjust number elements of this row
                    if (i >= lhs_offset) {
                        lhs_sublist_length += lhs_metadata[i].length;
                    // Adjust number elements of previous rows
                    } else if (i < lhs_offset) {
                        lhs_prev_length += lhs_metadata[i].length;
                    }
                    // Adjust number elements of this row
                    if (j >= rhs_offset) {
                        rhs_sublist_length += rhs_metadata[j].length;
                    // Adjust number elements of previous rows
                    } else if (j < rhs_offset) {
                        rhs_prev_length += rhs_metadata[j].length;
                    }
                }
                current_lhs_vec = &ListVector::GetEntry(*current_lhs_vec);
                current_rhs_vec = &ListVector::GetEntry(*current_rhs_vec);
                // Adjust offset for next dimension
                lhs_offset = lhs_prev_length;
                rhs_offset = rhs_prev_length;
                // Adjust length for next dimension
                lhs_length = lhs_sublist_length;
                rhs_length = rhs_sublist_length;
            }

            auto result_data = FlatVector::GetData<TYPE>(*result_child);

            // Perform the actual list operation 
            OP::Operation(lhs_data + lhs_offset, rhs_data + rhs_offset, result_data + result_offset, lhs_length);
            result_offset += lhs_length;
            return result_list;
        });

    if (args.AllConstant()) {
        result.SetVectorType(VectorType::CONSTANT_VECTOR);
    }
    ListVector::SetListSize(result, current_size);
}

/**
 * This function executes matrix multiplication on lists
 * (IMPORTANT: Limited on lists with at most two dimensions)
 */
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

    // Select the child vectors which are not of type LIST (vectors which contain elements of type TYPE)
    auto lhs_list_size = ListVector::GetListSize(lhs_vec);
    auto rhs_list_size = ListVector::GetListSize(rhs_vec);
    auto *lhs_child = &ListVector::GetEntry(lhs_vec);
    auto *rhs_child = &ListVector::GetEntry(rhs_vec);
    auto *result_child = &ListVector::GetEntry(result);
    while(lhs_child->GetType().id() == LogicalTypeId::LIST) {
        lhs_list_size = ListVector::GetListSize(*lhs_child);
        lhs_child = &ListVector::GetEntry(*lhs_child);
    }
    while(rhs_child->GetType().id() == LogicalTypeId::LIST) {
        rhs_list_size = ListVector::GetListSize(*rhs_child);
        rhs_child = &ListVector::GetEntry(*rhs_child);
        result_child = &ListVector::GetEntry(*result_child);
    }

    // Transform ListVector into FlatVector to get access to the elements
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
    // Start index of data (jump to values that corresponds to a specific row)
    idx_t result_data_offset = 0;
    
    // Function that will be executed for each row
    BinaryExecutor::ExecuteWithNulls<list_entry_t, list_entry_t, list_entry_t>(
        lhs_vec, rhs_vec, result, count,
        [&](const list_entry_t &left, const list_entry_t &right, ValidityMask &mask, idx_t row_idx) {
            // Dimension structure of each matrix
            auto rowsA = left.length;
            auto rowsB = right.length;
            uint64_t colsA = 1;
            uint64_t colsB = 1;
            // Offset to the data of type TYPE which corresponds to the current selected LIST
            idx_t prev_lhs_length = 0;
            idx_t prev_rhs_length = 0;

            // Reserve space for the result vector
            idx_t new_size = current_size + rowsA;
            ListVector::Reserve(result, new_size);
            // Set list metadata (of this row)
            list_entry_t result_metadata;
            result_metadata.offset = current_size;
            result_metadata.length = rowsA;

            // If the parameter vectors are empty, set the result to NULL
            if (!MatrixMultiplicationOperator::ALLOW_EMPTY && (left.length == 0 || right.length == 0)) {
                mask.SetInvalid(row_idx);
                return result_metadata;
            }

            // Identify the column dimension value of the left parameter if left has more than one dimension
            auto &left_child = ListVector::GetEntry(lhs_vec);
            // Extract column dimension from list_entry_t objects of child
            if (left_child.GetType().id() == LogicalTypeId::LIST){
                auto *metadata = ListVector::GetData(left_child);
                colsA = metadata[left.offset].length;
                for(idx_t i = 0; i < left.offset + left.length; i++) {
                    // Proof if all list_entry_t objects have the same structure of the current row
                    if (colsA != metadata[i].length && i >= left.offset) {
                        throw InvalidInputException("Left list has an unevenly distributed number of elements");
                    }
                    // Count the number of elements of previous rows
                    if (i < left.offset) {
                        prev_lhs_length += metadata[i].length;
                    }
                }
            }

            // Identify the column dimension value of the right parameter if right has more than one dimension
            auto &right_child = ListVector::GetEntry(rhs_vec);
            // Extract column dimension from list_entry_t objects of child
            if (right_child.GetType().id() == LogicalTypeId::LIST) {
                auto *metadata = ListVector::GetData(right_child);
                colsB = metadata[right.offset].length;
                for(idx_t i = 0; i < right.offset + right.length; i++) {
                    // Proof if all list_entry_t objects have the same structure of the current row
                    if (colsB != metadata[i].length && i >= right.offset) {
                        throw InvalidInputException("Right list has an unevenly distributed number of elements");
                    }
                    // Count the number of elements of previous rows
                    if (i < right.offset) {
                        prev_rhs_length += metadata[i].length;
                    }
                }
            }

            // Check if the dimensions are valid for matrix multiplication
            if (colsA != rowsB) {
                throw InvalidInputException(
                    "%s: invalid dimension structure for matrix multiplication, got '%d'x'%d' and '%d'x'%d'", func_name,
                    rowsA, colsA, rowsB, colsB);
            }

            // Append sublist
            Vector subvec(duckdb::LogicalType::LIST(rhs_child->GetType()));
            ListVector::Reserve(subvec, rowsA * colsB);
            ListVector::SetListSize(subvec, rowsA * colsB);
            auto* list_data = ListVector::GetData(subvec);
            for (idx_t i = 0; i < rowsA; i++) {
                list_data[i].offset = i * colsB;
                list_data[i].length = colsB;
            }
            ListVector::Append(result, subvec, rowsA);
            // Get shared pointer to actual data in result
            auto result_data = FlatVector::GetData<TYPE>(*result_child);

            // Perform the actual multiplication operation
            MatrixMultiplicationOperator::Operation(
                lhs_data + prev_lhs_length, 
                rhs_data + prev_rhs_length,
                result_data + result_data_offset,
                rowsA,
                rowsB,
                colsB
            );
            // Adjust result size and data offset
            current_size += result_metadata.length; 
            result_data_offset += rowsA * colsB;
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

/**
 * This struct stores important properties to select the correct function
 */
struct ListArithBindData : public FunctionData {
    LogicalType element_type;
    bool scalar_op;
    ListArithBindData(LogicalType element_type, bool scalar_op) : element_type(element_type), scalar_op(scalar_op) {}
    unique_ptr<FunctionData> Copy() const override { 
        return make_uniq<ListArithBindData>(element_type, scalar_op); 
    }
    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<ListArithBindData>();
        return element_type == other.element_type && scalar_op == other.scalar_op;
    }
};

/**
 * This function determines if the given parameters are valid and selects the return type
 */
static unique_ptr<FunctionData> ListArithBind(ClientContext &, ScalarFunction &bound_function, vector<unique_ptr<Expression>> &arguments) {
    D_ASSERT(arguments.size() == 2);
    LogicalType element_type;
    bool scalar_op = false;
    LogicalType arg1_type = arguments[0]->return_type;
    LogicalType arg2_type = arguments[1]->return_type;
    if (arg1_type.id() != LogicalTypeId::LIST && !arg1_type.IsNumeric()) {
        throw BinderException("%s is not supported for this function", arg1_type);
    }
    if (arg2_type.id() != LogicalTypeId::LIST && !arg2_type.IsNumeric()) {
        throw BinderException("%s is not supported for this function", arg2_type);
    }
    if (arg1_type.id() != LogicalTypeId::LIST && arg2_type.id() != LogicalTypeId::LIST) {
        throw BinderException("At least one argument must be a LIST");
    }
    // If the input parameters are two lists
    if (arg1_type.id() == LogicalTypeId::LIST && arg2_type.id() == LogicalTypeId::LIST) {
        while(arg1_type.id() == LogicalTypeId::LIST) {
            arg1_type = ListType::GetChildType(arg1_type);
            arg2_type = ListType::GetChildType(arg2_type);
            if (arg1_type != arg2_type) {
                throw BinderException("Unequal types %s and %s", arg1_type, arg2_type);
            }
        }
        element_type = arg1_type;
        bound_function.return_type = arguments[0]->return_type;
    }
    // If left is list and right a scalar
    if (arg1_type.id() == LogicalTypeId::LIST) {
        while(arg1_type.id() == LogicalTypeId::LIST) {
            arg1_type = ListType::GetChildType(arg1_type);
        }
        element_type = arg1_type;
        scalar_op = true;
        bound_function.return_type = arguments[0]->return_type;
        if (!arg2_type.IsNumeric()) {
            throw BinderException("Provided scalar is not numeric");
        }
        bound_function.arguments[1] = arg1_type;
    }
    // If left is scalar and right a list
    if (arg2_type.id() == LogicalTypeId::LIST) {
        while(arg2_type.id() == LogicalTypeId::LIST) {
            arg2_type = ListType::GetChildType(arg2_type);
        }
        element_type = arg2_type;
        scalar_op = true;
        bound_function.return_type = arguments[1]->return_type;
        if (!arg1_type.IsNumeric()) {
            throw BinderException("Provided scalar is not numeric");
        }
        bound_function.arguments[0] = arg2_type;
    }
    return make_uniq<ListArithBindData>(element_type, scalar_op);
}

/**
 * This function selects the correct function according to the parameter types
 */
template<class OP>
static void ListArithExec(DataChunk &args, ExpressionState &state, Vector &result) {
    auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
    auto &info = func_expr.bind_info->Cast<ListArithBindData>();
    switch (info.element_type.id()) {
    case LogicalTypeId::FLOAT:
        if (info.scalar_op) {
            ListGenericArithScalar<float, OP>(args, state, result);
        } else {
            ListGenericArithList<float, OP>(args, state, result);
        } 
        break;
    case LogicalTypeId::DOUBLE:
        if (info.scalar_op) {
            ListGenericArithScalar<double, OP>(args, state, result);
        } else {
            ListGenericArithList<double, OP>(args, state, result);
        }
        break;
    case LogicalTypeId::BFLOAT:
        if (info.scalar_op) {
            ListGenericArithScalar<std::bfloat16_t, OP>(args, state, result);
        } else {
            ListGenericArithList<std::bfloat16_t, OP>(args, state, result);
        }
        break;
    default:
        throw NotImplementedException("Unsupported element type for list arithmetic");
    }
}

/**
 * Registers the list_add() functions
 */
ScalarFunctionSet ListArithAddFun::GetFunctions() {
	ScalarFunctionSet set("list_add");
	set.AddFunction(ScalarFunction({
                        LogicalType::LIST(LogicalType::ANY), 
                        LogicalType::LIST(LogicalType::ANY)}, 
                        LogicalType::LIST(LogicalType::ANY), 
                        ListArithExec<AddOperator>,
	                    ListArithBind));
    set.AddFunction(ScalarFunction({
                        LogicalType::ANY, 
                        LogicalType::LIST(LogicalType::ANY)}, 
                        LogicalType::LIST(LogicalType::ANY), 
                        ListArithExec<AddOperator>,
	                    ListArithBind));
    set.AddFunction(ScalarFunction({
                        LogicalType::LIST(LogicalType::ANY), 
                        LogicalType::ANY}, 
                        LogicalType::LIST(LogicalType::ANY), 
                        ListArithExec<AddOperator>,
	                    ListArithBind));
	for (auto &func : set.functions) {
		BaseScalarFunction::SetReturnsError(func);
	}
	return set;
}

/**
 * Registers the list_sub() functions
 */
ScalarFunctionSet ListArithSubFun::GetFunctions() {
	ScalarFunctionSet set("list_sub");
	set.AddFunction(ScalarFunction({
                        LogicalType::LIST(LogicalType::ANY), 
                        LogicalType::LIST(LogicalType::ANY)}, 
                        LogicalType::LIST(LogicalType::ANY), 
                        ListArithExec<SubOperator>,
	                    ListArithBind));
    set.AddFunction(ScalarFunction({
                        LogicalType::ANY, 
                        LogicalType::LIST(LogicalType::ANY)}, 
                        LogicalType::LIST(LogicalType::ANY), 
                        ListArithExec<SubOperator>,
	                    ListArithBind));
    set.AddFunction(ScalarFunction({
                        LogicalType::LIST(LogicalType::ANY), 
                        LogicalType::ANY}, 
                        LogicalType::LIST(LogicalType::ANY), 
                        ListArithExec<SubOperator>,
	                    ListArithBind));
	for (auto &func : set.functions) {
		BaseScalarFunction::SetReturnsError(func);
	}
	return set;
}

/**
 * Registers the list_mul() functions
 */
ScalarFunctionSet ListArithMulFun::GetFunctions() {
	ScalarFunctionSet set("list_mul");
	set.AddFunction(ScalarFunction({
                        LogicalType::LIST(LogicalType::ANY), 
                        LogicalType::LIST(LogicalType::ANY)}, 
                        LogicalType::LIST(LogicalType::ANY), 
                        ListArithExec<MulOperator>,
	                    ListArithBind));
    set.AddFunction(ScalarFunction({
                        LogicalType::ANY, 
                        LogicalType::LIST(LogicalType::ANY)}, 
                        LogicalType::LIST(LogicalType::ANY), 
                        ListArithExec<MulOperator>,
	                    ListArithBind));
    set.AddFunction(ScalarFunction({
                        LogicalType::LIST(LogicalType::ANY), 
                        LogicalType::ANY}, 
                        LogicalType::LIST(LogicalType::ANY), 
                        ListArithExec<MulOperator>,
	                    ListArithBind));
	for (auto &func : set.functions) {
		BaseScalarFunction::SetReturnsError(func);
	}
	return set;
}

/**
 * Registers the list_div() functions
 */
ScalarFunctionSet ListArithDivFun::GetFunctions() {
	ScalarFunctionSet set("list_div");
	set.AddFunction(ScalarFunction({
                        LogicalType::LIST(LogicalType::ANY), 
                        LogicalType::LIST(LogicalType::ANY)}, 
                        LogicalType::LIST(LogicalType::ANY), 
                        ListArithExec<DivOperator>,
	                    ListArithBind));
    set.AddFunction(ScalarFunction({
                        LogicalType::ANY, 
                        LogicalType::LIST(LogicalType::ANY)}, 
                        LogicalType::LIST(LogicalType::ANY), 
                        ListArithExec<DivOperator>,
	                    ListArithBind));
    set.AddFunction(ScalarFunction({
                        LogicalType::LIST(LogicalType::ANY), 
                        LogicalType::ANY}, 
                        LogicalType::LIST(LogicalType::ANY), 
                        ListArithExec<DivOperator>,
	                    ListArithBind));
	for (auto &func : set.functions) {
		BaseScalarFunction::SetReturnsError(func);
	}
	return set;
}

/**
 * Registers the list_mmul() functions
 */
ScalarFunctionSet ListArithMMulFun::GetFunctions() {
	ScalarFunctionSet set("list_mmul");
	for (auto &type : LogicalType::Real()) {
        const auto list_single = LogicalType::LIST(type);
        const auto list_double = LogicalType::LIST(LogicalType::LIST(type));
        if (type.id() == LogicalTypeId::FLOAT) {
            set.AddFunction(ScalarFunction({list_double, list_double}, list_double, ListMatrixMul<float>));
            set.AddFunction(ScalarFunction({list_double, list_single}, list_double, ListMatrixMul<float>));
        } else if (type.id() == LogicalTypeId::BFLOAT) {
            set.AddFunction(ScalarFunction({list_double, list_double}, list_double, ListMatrixMul<std::bfloat16_t>));
            set.AddFunction(ScalarFunction({list_double, list_single}, list_double, ListMatrixMul<std::bfloat16_t>));
        } else if (type.id() == LogicalTypeId::DOUBLE) {
            set.AddFunction(ScalarFunction({list_double, list_double}, list_double, ListMatrixMul<double>));
            set.AddFunction(ScalarFunction({list_double, list_single}, list_double, ListMatrixMul<double>));
        }
	}
	for (auto &func : set.functions) {
		BaseScalarFunction::SetReturnsError(func);
	}
	return set;
}
}