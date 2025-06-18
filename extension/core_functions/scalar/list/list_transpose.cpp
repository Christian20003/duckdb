#include "core_functions/scalar/list_functions.hpp"
#include "core_functions/array_kernels.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

#include <stdfloat>

namespace duckdb
{

template <class TYPE>
static void ListTransposeFun(DataChunk &args, ExpressionState &state, Vector &result) {
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
    // Start index of list_entry_t objects (jump to the first entry for a specific row)
    idx_t start_idx = 0;
    // Start index of data (jump to values that corresponds to a specific row)
    idx_t offset = 0;
    idx_t result_offset = 0;
    
    // Function that will be executed for each row
    UnaryExecutor::ExecuteWithNulls<list_entry_t, list_entry_t>(
        vector, result, count,
        [&](const list_entry_t &list, ValidityMask &mask, idx_t row_idx) {
            // Dimensions that should be transposed
            auto rows = list.length;
            uint64_t cols = 0;

            // Extract cols dimension
            auto &child = ListVector::GetEntry(vector);
            // If list has more than one dimension
            if (child.GetType().id() == LogicalTypeId::LIST) {
                // Get list_entry_t objects of current child vector
                auto metadata = ListVector::GetData(child);
                // Get the size specification and proof if it match with all list_entry_t objects of the same row
                auto start = start_idx;
                auto condition = start_idx + list.length;
                for(idx_t i = start; i < condition; i++) {
                    if (cols == 0) {
                        cols = metadata[i].length;
                    }
                    if (cols != metadata[i].length) {
                        throw InvalidInputException("List has an unevenly distributed number of elements");
                    }
                }
            // If list has only one dimension
            } else {
                cols = 1;
            }
            
            // Reserve space for the result vector
            idx_t new_size = current_size + cols;
            ListVector::Reserve(result, new_size);
            // Set list metadata (of this row)
            list_entry_t result_metadata;
            result_metadata.offset = current_size;
            result_metadata.length = cols;

            // Create subvector which contains result of this row
            Vector subvec(duckdb::LogicalType::LIST(vec_child->GetType()));
            ListVector::Reserve(subvec, cols * rows);
            ListVector::SetListSize(subvec, cols * rows);
            auto* list_data = ListVector::GetData(subvec);
            for (idx_t i = 0; i < cols; i++) {
                list_data[i].offset = i * rows;
                list_data[i].length = rows;
            }
            ListVector::Append(result, subvec, cols);

            // Needed if input is one dimensional and output two dimensional
            if (result_child->GetType().id() == LogicalTypeId::LIST) {
                result_child = &ListVector::GetEntry(*result_child);
            }
            auto result_data = FlatVector::GetData<TYPE>(*result_child);
            
            // If the parameter vectors are empty, set the result vector to NULL
            if (!TransposeOperator::ALLOW_EMPTY && list.length == 0) {
                mask.SetInvalid(row_idx);
                return result_metadata;
            }

            // Perform the actual transpose operation
            TransposeOperator::Operation(
                vec_data + offset, 
                result_data + result_offset,
                rows,
                cols
            );
            // Adjust control variables
            current_size += result_metadata.length; 
            start_idx += list.length;
            offset += rows * cols;
            result_offset += rows * cols;
            return result_metadata;
        });

    if (args.AllConstant()) {
        result.SetVectorType(VectorType::CONSTANT_VECTOR);
    }
    ListVector::SetListSize(result, current_size);
}

ScalarFunctionSet ListTranspose::GetFunctions() {
	ScalarFunctionSet set("transpose");
	for (auto &type : LogicalType::Real()) {
        const auto list_single = LogicalType::LIST(type);
        const auto list_double = LogicalType::LIST(LogicalType::LIST(type));
        if (type.id() == LogicalTypeId::FLOAT) {
            set.AddFunction(ScalarFunction({list_single}, list_double, ListTransposeFun<float>));
            set.AddFunction(ScalarFunction({list_double}, list_double, ListTransposeFun<float>));
        } else if (type.id() == LogicalTypeId::BFLOAT) {
            set.AddFunction(ScalarFunction({list_single}, list_double, ListTransposeFun<std::bfloat16_t>));
            set.AddFunction(ScalarFunction({list_double}, list_double, ListTransposeFun<std::bfloat16_t>));
        } else if (type.id() == LogicalTypeId::DOUBLE) {
            set.AddFunction(ScalarFunction({list_single}, list_double, ListTransposeFun<double>));
            set.AddFunction(ScalarFunction({list_double}, list_double, ListTransposeFun<double>));
        }
	}
	for (auto &func : set.functions) {
		BaseScalarFunction::SetReturnsError(func);
	}
	return set;
}
} // namespace duckdb
