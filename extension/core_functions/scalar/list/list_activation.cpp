#include "core_functions/scalar/list_functions.hpp"
#include "core_functions/array_kernels.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

#include <stdfloat>

namespace duckdb
{

template <class TYPE, class OP>
static void ListActivationFun(DataChunk &args, ExpressionState &state, Vector &result) {
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
    auto *result_child = &ListVector::GetEntry(result);
    while(vec_child->GetType().id() == LogicalTypeId::LIST) {
        vec_size = ListVector::GetListSize(*vec_child);
        vec_child = &ListVector::GetEntry(*vec_child);
        result_child = &ListVector::GetEntry(*result_child);
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
    
    // Stores at the end the overall size of the resulting vector
    auto current_size = ListVector::GetListSize(result);
    // Start index of list_entry_t objects (jump to the start entry for a specific row)
    idx_t start_idx = 0;
    // Start index of data (jump to values that corresponds to a specific row)
    idx_t offset = 0;
    idx_t result_offset = 0;
    
    // Function that will be executed for each row
    UnaryExecutor::ExecuteWithNulls<list_entry_t, list_entry_t>(
        vector, result, count,
        [&](const list_entry_t &list, ValidityMask &mask, idx_t row_idx) {
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
                auto start = start_idx;
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

            // Set list metadata (of this row)
            list_entry_t result_metadata;
            result_metadata.offset = current_size;
            result_metadata.length = list.length;

            // Get a pointer to the first element of result
            auto result_data = FlatVector::GetData<TYPE>(*result_child);
            
            // If the parameter vectors are empty, set the result vector to NULL
            if (!OP::ALLOW_EMPTY && list.length == 0) {
                mask.SetInvalid(row_idx);
                return result_metadata;
            }

            // Perform the actual activation operation
            OP::Operation(
                vec_data + offset, 
                result_data + result_offset,
                number_elements
            );
            // Adjust control variables
            current_size += result_metadata.length; 
            start_idx += list.length;
            offset += number_elements;
            result_offset += number_elements;
            return result_metadata;
        });

    if (args.AllConstant()) {
        result.SetVectorType(VectorType::CONSTANT_VECTOR);
    }
    ListVector::SetListSize(result, current_size);
}

ScalarFunctionSet ListSigmoid::GetFunctions() {
	ScalarFunctionSet set("sig");
	for (auto &type : LogicalType::Real()) {
        const auto list_single = LogicalType::LIST(type);
        const auto list_double = LogicalType::LIST(LogicalType::LIST(type));
        if (type.id() == LogicalTypeId::FLOAT) {
            set.AddFunction(ScalarFunction({list_single}, list_single, ListActivationFun<float, SigmoidOperator>));
            set.AddFunction(ScalarFunction({list_double}, list_double, ListActivationFun<float, SigmoidOperator>));
        } else if (type.id() == LogicalTypeId::BFLOAT) {
            set.AddFunction(ScalarFunction({list_single}, list_single, ListActivationFun<std::bfloat16_t, SigmoidOperator>));
            set.AddFunction(ScalarFunction({list_double}, list_double, ListActivationFun<std::bfloat16_t, SigmoidOperator>));
        } else if (type.id() == LogicalTypeId::DOUBLE) {
            set.AddFunction(ScalarFunction({list_single}, list_single, ListActivationFun<double, SigmoidOperator>));
            set.AddFunction(ScalarFunction({list_double}, list_double, ListActivationFun<double, SigmoidOperator>));
        }
	}
	for (auto &func : set.functions) {
		BaseScalarFunction::SetReturnsError(func);
	}
	return set;
}
} // namespace duckdb