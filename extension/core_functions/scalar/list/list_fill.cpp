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
    auto &vec = args.data[1];

    // Get list size
    auto vec_size = ListVector::GetListSize(vec);

    // Get child vector
    auto *vec_child = &ListVector::GetEntry(vec);
    auto *result_child = &ListVector::GetEntry(result);

    // Decompress the list vector (with single values) and flatten them
    vec_child->Flatten(vec_size);

    D_ASSERT(vec_child->GetVectorType() == VectorType::FLAT_VECTOR);

    // NULL values are not allowed
    if (!FlatVector::Validity(*vec_child).CheckAllValid(vec_size)) {
        throw InvalidInputException("%s: argument can not contain NULL values", func_name);
    }

    // Get the actual data as shared pointer to the first element
    auto vec_data = FlatVector::GetData<int32_t>(*vec_child);
    
    // Create control variables
    auto current_size = ListVector::GetListSize(result);
    // Start index of data (jump to values that corresponds to a specific row)
    idx_t offset = 0;
    idx_t result_offset = 0;
    
    // Function that will be executed for each row
    BinaryExecutor::ExecuteWithNulls<TYPE, list_entry_t, list_entry_t>(
        value, vec, result, count,
        [&](const TYPE content, const list_entry_t &dimensions, ValidityMask &mask, idx_t row_idx) {
            if (dimensions.length > 2) {
                throw InvalidInputException("%s: Only lists with at most 2 values are supported", func_name);
            } else if (dimensions.length == 0) {
                throw InvalidInputException("%s: List requires at least one value", func_name);
            }

            idx_t rows = *(vec_data + offset);
            idx_t cols = dimensions.length == 2 ? *(vec_data + 1 + offset) : 0;

            // Reserve space for the result vector
            idx_t new_size = current_size + rows;
            ListVector::Reserve(result, new_size);
            // Set list metadata (of this row)
            list_entry_t result_metadata;
            result_metadata.offset = current_size;
            result_metadata.length = rows;

            if (cols != 0) {
                for (idx_t i = 0; i < rows; i++) {
                    Vector subvec(duckdb::LogicalType::LIST(vec_child->GetType()));
                    ListVector::Reserve(subvec, cols);
                    ListVector::SetListSize(subvec, cols);
                    auto* list_data = ListVector::GetData(subvec);
                    list_data->offset = 0;
                    list_data->length = cols;
                    ListVector::Append(result, subvec, 1);
                }
            }
            // Get shared pointer to actual data
            auto result_data = FlatVector::GetData<TYPE>(*result_child);
            
            TYPE *result_ptr = result_data + result_offset;
            auto condition = cols != 0 ? rows * cols : rows;
            for(idx_t i = 0; i < condition; i++) {
                *result_ptr++ = content;
            }

            // Adjust control variable
            current_size += result_metadata.length; 
            result_offset += cols != 0 ? rows * cols : rows;
            offset += dimensions.length; 
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
        const auto list_double = LogicalType::LIST(LogicalType::LIST(type));
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
