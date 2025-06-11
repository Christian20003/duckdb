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

    // Get function parameters (IMPORTANT: This will include all rows from a chunk)
    auto &vec = args.data[0];

    // Get list size
    auto vec_size = ListVector::GetListSize(vec);

    // Get child vector
    auto *vec_child = &ListVector::GetEntry(vec);
    auto *result_child = &ListVector::GetEntry(result);

    // If the current child vectors contain further lists, select their children until reaching last level
    // And extract their list size
    while(vec_child->GetType().id() == LogicalTypeId::LIST) {
        vec_size = ListVector::GetListSize(*vec_child);
        vec_child = &ListVector::GetEntry(*vec_child);
        result_child = &ListVector::GetEntry(*result_child);
    }

    // Decompress the list vector (with single values) and flatten them
    vec_child->Flatten(vec_size);

    D_ASSERT(vec_child->GetVectorType() == VectorType::FLAT_VECTOR);

    // NULL values are not allowed
    if (!FlatVector::Validity(*vec_child).CheckAllValid(vec_size)) {
        throw InvalidInputException("%s: argument can not contain NULL values", func_name);
    }

    // Get the actual data as shared pointer to the first element
    auto vec_data = FlatVector::GetData<TYPE>(*vec_child);
    
    // Create control variables
    auto current_size = ListVector::GetListSize(result);
    // Start index of metadata lists (jump to the entry for a specific row)
    idx_t start_index = 0;
    // Start index of data (jump to values that corresponds to a specific row)
    idx_t offset = 0;
    idx_t result_offset = 0;
    
    // Function that will be executed for each row
    UnaryExecutor::ExecuteWithNulls<list_entry_t, list_entry_t>(
        vec, result, count,
        [&](const list_entry_t &param, ValidityMask &mask, idx_t row_idx) {
            // Extract dimension values
            auto rows = param.length;
            uint64_t cols = 0;

            // If left is multi-dimensional get list metadata of each sublist that contains single elements
            auto &child = ListVector::GetEntry(vec);
            auto metadata = FlatVector::GetData<list_entry_t>(child);
            // If vector is constant ignore adjusting to the corresponding row
            auto start = start_index;
            auto condition = start_index + param.length;
            for(idx_t i = start; i < condition; i++) {
                // Get the size specification and proof if it match with all lists on the same level
                if (cols == 0) {
                    cols = metadata[i].length;
                }
                if (cols != metadata[i].length) {
                    throw InvalidInputException("List has an unevenly distributed number of elements");
                }
            }

            // Reserve space for the result vector
            idx_t new_size = current_size + cols;
            ListVector::Reserve(result, new_size);
            // Set list metadata (of this row)
            list_entry_t result_metadata;
            result_metadata.offset = current_size;
            result_metadata.length = cols;

            for (idx_t i = 0; i < rows; i++) {
                Vector subvec(duckdb::LogicalType::LIST(vec_child->GetType()));
                ListVector::Reserve(subvec, cols);
                ListVector::SetListSize(subvec, cols);
                auto* list_data = ListVector::GetData(subvec);
                list_data->offset = 0;
                list_data->length = cols;
                ListVector::Append(result, subvec, 1);
            }
            // Get shared pointer to actual data
            auto result_data = FlatVector::GetData<TYPE>(*result_child);
            
            // If the parameter vectors are empty, set the result vector to NULL
            if (!OP::ALLOW_EMPTY && param.length == 0) {
                mask.SetInvalid(row_idx);
                return result_metadata;
            }

            // Perform the actual addition operation
            OP::Operation(
                vec_data + offset, 
                result_data + result_offset,
                rows * cols
            );
            // Adjust control variable
            current_size += result_metadata.length; 
            start_index += param.length;
            offset += rows * cols;
            result_offset += rows * cols;
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
        // const auto list_single = LogicalType::LIST(type);
        const auto list_double = LogicalType::LIST(LogicalType::LIST(type));
        if (type.id() == LogicalTypeId::FLOAT) {
            set.AddFunction(ScalarFunction({list_double}, list_double, ListActivationFun<float, SigmoidOperator>));
        } else if (type.id() == LogicalTypeId::BFLOAT) {
            set.AddFunction(ScalarFunction({list_double}, list_double, ListActivationFun<std::bfloat16_t, SigmoidOperator>));
        } else if (type.id() == LogicalTypeId::DOUBLE) {
            set.AddFunction(ScalarFunction({list_double}, list_double, ListActivationFun<double, SigmoidOperator>));
        }
	}
	for (auto &func : set.functions) {
		BaseScalarFunction::SetReturnsError(func);
	}
	return set;
}
} // namespace duckdb