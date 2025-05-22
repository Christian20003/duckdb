#include "core_functions/scalar/list_functions.hpp"
#include "core_functions/array_kernels.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

#include <stdfloat>

namespace duckdb {

template <class TYPE, class OP>
static void ListGenericArith(DataChunk &args, ExpressionState &state, Vector &result) {
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
    auto result_data = FlatVector::GetData<TYPE>(*result_child);
    
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
            // Reserve space for the result vector and copy all elements from the left vector into result
            idx_t new_size = current_size + left.length;
            ListVector::Reserve(result, new_size);
            VectorOperations::Copy(ListVector::GetEntry(lhs_vec), ListVector::GetEntry(result), left.offset + left.length, left.offset, current_size);
            
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
//-------------------------------------------------------------------------
// Function Registration
//-------------------------------------------------------------------------

template <class OP>
static void AddListArithFunction(ScalarFunctionSet &set, const LogicalType &type) {
	const auto list_single = LogicalType::LIST(type);
    const auto list_double = LogicalType::LIST(LogicalType::LIST(type));
	if (type.id() == LogicalTypeId::FLOAT) {
		set.AddFunction(ScalarFunction({list_single, list_single}, list_single, ListGenericArith<float, OP>));
        set.AddFunction(ScalarFunction({list_double, list_double}, list_double, ListGenericArith<float, OP>));
	} else if (type.id() == LogicalTypeId::BFLOAT) {
		set.AddFunction(ScalarFunction({list_single, list_single}, list_single, ListGenericArith<std::bfloat16_t, OP>));
        set.AddFunction(ScalarFunction({list_double, list_double}, list_double, ListGenericArith<std::bfloat16_t, OP>));
	} else if (type.id() == LogicalTypeId::DOUBLE) {
		set.AddFunction(ScalarFunction({list_single, list_single}, list_single, ListGenericArith<double, OP>));
        set.AddFunction(ScalarFunction({list_double, list_double}, list_double, ListGenericArith<double, OP>));
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
}