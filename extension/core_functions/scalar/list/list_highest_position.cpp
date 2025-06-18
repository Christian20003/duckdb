#include "core_functions/scalar/list_functions.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

#include <stdfloat>

namespace duckdb
{

template <class TYPE>
static void HighestPositionFun(DataChunk &args, ExpressionState &state, Vector &result) {
    // Extract function name
    const auto &lstate = state.Cast<ExecuteFunctionState>();
    const auto &expr = lstate.expr.Cast<BoundFunctionExpression>();
    const auto &func_name = expr.function.name;

    // Get number of rows
    auto count = args.size();

    // Get function parameter (IMPORTANT: This will include all rows from a chunk)
    auto &vector = args.data[0];

    // Select the child vector which is not of type LIST (vector which contains elements of type TYPE)
    auto vec_size = ListVector::GetListSize(vector);
    auto *vec_child = &ListVector::GetEntry(vector);
    while(vec_child->GetType().id() == LogicalTypeId::LIST) {
        vec_size = ListVector::GetListSize(*vec_child);
        vec_child = &ListVector::GetEntry(*vec_child);
    }

    // Transform ListVector into FlatVector to get access to the elements
    vec_child->Flatten(vec_size);
    D_ASSERT(vec_child->GetVectorType() == VectorType::FLAT_VECTOR);

    // NULL values are not allowed
    if (!FlatVector::Validity(*vec_child).CheckAllValid(vec_size)) {
        throw InvalidInputException("%s: argument can not contain NULL values", func_name);
    }

    // Get a pointer to the first element
    auto vec_data = FlatVector::GetData<TYPE>(*vec_child);
    
    // Function that will be executed for each row
    UnaryExecutor::ExecuteWithNulls<list_entry_t, int32_t>(
        vector, result, count,
        [&](const list_entry_t &list, ValidityMask &mask, idx_t row_idx) {
            // If the parameter vector is empty, set the result to NULL
            if (list.length == 0) {
                mask.SetInvalid(row_idx);
                return 0;
            }
            // Value which stores the current offset of a specific dimension
            idx_t offset = list.offset;
            // Value which store the current length of a specific dimension
            idx_t length = list.length;

            auto *child = &ListVector::GetEntry(vector);
            while(child->GetType().id() == LogicalTypeId::LIST) {
                // Value which stores the number of elements in the current dimension and row
                idx_t sublist_length = 0;
                // Value which stores the number of elements in the current dimension and previous rows
                idx_t prev_length = 0;
                // Get list_entry_t objects of current child vector
                auto *metadata = ListVector::GetData(*child);
                for(idx_t i = 0; i < offset + length; i++) {
                    // Adjust number elements of this row
                    if (i >= offset) {
                        sublist_length += metadata[i].length;
                    // Adjust number elements of previous rows
                    } else {
                        prev_length += metadata[i].length;
                    }
                }
                // Adjust offset for next dimension
                offset = prev_length;
                // Adjust length for next dimension
                length = sublist_length;
                child = &ListVector::GetEntry(*child);
            }

            // Find index with largest value
            int32_t index = 0;
            TYPE value = *(vec_data + offset);
            for(idx_t i = 0; i < length; i++) {
                if(value < *(vec_data + offset + i)) {
                    index = i;
                    value = *(vec_data + offset + i);
                }
            }

            return index;
        });

    if (args.AllConstant()) {
        result.SetVectorType(VectorType::CONSTANT_VECTOR);
    }
}

ScalarFunctionSet HighestPosition::GetFunctions() {
	ScalarFunctionSet set("highestposition");
	for (auto &type : LogicalType::Real()) {
        const auto list_single = LogicalType::LIST(type);
        const auto list_double = LogicalType::LIST(LogicalType::LIST(type));
        if (type.id() == LogicalTypeId::FLOAT) {
            set.AddFunction(ScalarFunction({list_single}, LogicalType::UINTEGER, HighestPositionFun<float>));
            set.AddFunction(ScalarFunction({list_double}, LogicalType::UINTEGER, HighestPositionFun<float>));
        } else if (type.id() == LogicalTypeId::BFLOAT) {
            set.AddFunction(ScalarFunction({list_single}, LogicalType::UINTEGER, HighestPositionFun<std::bfloat16_t>));
            set.AddFunction(ScalarFunction({list_double}, LogicalType::UINTEGER, HighestPositionFun<std::bfloat16_t>));
        } else if (type.id() == LogicalTypeId::DOUBLE) {
            set.AddFunction(ScalarFunction({list_single}, LogicalType::UINTEGER, HighestPositionFun<double>));
            set.AddFunction(ScalarFunction({list_double}, LogicalType::UINTEGER, HighestPositionFun<double>));
        }
	}
	for (auto &func : set.functions) {
		BaseScalarFunction::SetReturnsError(func);
	}
	return set;
}
} // namespace duckdb
