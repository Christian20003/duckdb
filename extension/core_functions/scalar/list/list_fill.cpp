#include "core_functions/scalar/list_functions.hpp"
#include "core_functions/array_kernels.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

#include <stdfloat>

namespace duckdb
{

template <class TYPE>
static void ListFillFun(DataChunk &args, ExpressionState &state, Vector &result) {
    // Extract function name
    const auto &lstate = state.Cast<ExecuteFunctionState>();
    const auto &expr = lstate.expr.Cast<BoundFunctionExpression>();
    const auto &func_name = expr.function.name;

    // Get number of rows
    auto count = args.size();

    // Get function parameters (IMPORTANT: This will include all rows from a chunk)
    auto &value = args.data[0];
    auto &vector = args.data[1];

    // Select the child vector which is not of type LIST (vector which contains elements of type INTEGER)
    auto vec_size = ListVector::GetListSize(vector);
    auto *vec_child = &ListVector::GetEntry(vector);

    // Transform ListVector into FlatVector to get access to the elements
    vec_child->Flatten(vec_size);
    D_ASSERT(vec_child->GetVectorType() == VectorType::FLAT_VECTOR);

    // NULL values are not allowed
    if (!FlatVector::Validity(*vec_child).CheckAllValid(vec_size)) {
        throw InvalidInputException("%s: argument can not contain NULL values", func_name);
    }

    // Get a pointer to the first element
    auto vec_data = FlatVector::GetData<int32_t>(*vec_child);

    // Get the number of result dimensions according to the return type
    // And the child vector of result which is not of type LIST (vector which contains elements of type TYPE)
    auto return_type = result.GetType();
    auto *result_child = &result;
    idx_t expected_dims = 0;
    while(return_type.id() == LogicalTypeId::LIST) {
        return_type = ListType::GetChildType(return_type);
        result_child = &ListVector::GetEntry(*result_child);
        expected_dims++;
    }
    
    // Stores at the end the overall size of the resulting vector
    auto current_size = ListVector::GetListSize(result);
    // Start index of data (jump to values that corresponds to a specific row)
    idx_t result_offset = 0;
    
    // Function that will be executed for each row
    BinaryExecutor::ExecuteWithNulls<TYPE, list_entry_t, list_entry_t>(
        value, vector, result, count,
        [&](const TYPE content, const list_entry_t &dimensions, ValidityMask &mask, idx_t row_idx) {
            if (dimensions.length != expected_dims) {
                throw InvalidInputException("%s: Expected a list with exactly %i entries", func_name, expected_dims);
            }

            // The current number of elements in a dimension
            idx_t dim_val = *(vec_data + dimensions.offset);

            // Reserve space for the result vector
            idx_t new_size = current_size + dim_val;
            ListVector::Reserve(result, new_size);
            // Set list metadata (of this row)
            list_entry_t result_metadata;
            result_metadata.offset = current_size;
            result_metadata.length = dim_val;

            // Pointer to vector which should get a child
            auto *to_append_vec = &result;
            auto type = result.GetType();
            idx_t number_elements = dim_val;
            // Iterate over each dimension (except the first one - is already assigned)
            for (idx_t i = 1; i < dimensions.length; i++) {
                // Build a new child vector which contains number of elements based on the given
                // value from the input vector
                dim_val = *(vec_data + dimensions.offset + i);
                type = ListType::GetChildType(type);
                Vector child(type);
                ListVector::Reserve(child, number_elements * dim_val);
                ListVector::SetListSize(child, number_elements * dim_val);
                auto* list_data = ListVector::GetData(child);
                for(idx_t j = 0; j < number_elements; j++) {
                    list_data[j].offset = j * dim_val;
                    list_data[j].length = dim_val;
                }
                // Append it to the upper vector
                ListVector::Append(*to_append_vec, child, number_elements);
                to_append_vec = &ListVector::GetEntry(*to_append_vec);
                number_elements *= dim_val;
            }
            // Get a pointer to the first element of result
            auto result_data = FlatVector::GetData<TYPE>(*result_child);
            
            // Add the input value to the result vector
            TYPE *result_ptr = result_data + result_offset;
            for(idx_t i = 0; i < number_elements; i++) {
                *result_ptr++ = content;
            }

            // Adjust control variable
            current_size += result_metadata.length; 
            result_offset += number_elements;
            return result_metadata;
        });

    if (args.AllConstant()) {
        result.SetVectorType(VectorType::CONSTANT_VECTOR);
    }
    ListVector::SetListSize(result, current_size);
}

ScalarFunctionSet ListFill::GetFunctions() {
	ScalarFunctionSet set("list_fill");
	for (auto &type : LogicalType::Real()) {
        const auto list_single = LogicalType::LIST(type);
        //const auto list_double = LogicalType::LIST(LogicalType::LIST(type));
        const auto list_arg = LogicalType::LIST(LogicalType::INTEGER);
        if (type.id() == LogicalTypeId::FLOAT) {
            set.AddFunction(ScalarFunction({type, list_arg}, list_single, ListFillFun<float>));
            //set.AddFunction(ScalarFunction({type, list_arg}, list_double, ListFillFun<float>));
        } else if (type.id() == LogicalTypeId::BFLOAT) {
            set.AddFunction(ScalarFunction({type, list_arg}, list_single, ListFillFun<std::bfloat16_t>));
            //set.AddFunction(ScalarFunction({type, list_arg}, list_double, ListFillFun<std::bfloat16_t>));
        } else if (type.id() == LogicalTypeId::DOUBLE) {
            set.AddFunction(ScalarFunction({type, list_arg}, list_single, ListFillFun<double>));
            //set.AddFunction(ScalarFunction({type, list_arg}, list_double, ListFillFun<double>));
        }
	}
	for (auto &func : set.functions) {
		BaseScalarFunction::SetReturnsError(func);
	}
	return set;
}
} // namespace duckdb
