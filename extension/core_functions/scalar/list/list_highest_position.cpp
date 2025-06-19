#include "core_functions/scalar/list_functions.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

#include <stdfloat>

namespace duckdb
{

/**
 * This function executes highest position on lists
 */
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

/**
 * This struct stores important properties to select the correct function
 */
struct ListHPBindData : public FunctionData {
    LogicalType element_type;

    ListHPBindData(LogicalType element_type) : element_type(element_type) {}
    unique_ptr<FunctionData> Copy() const override { 
        return make_uniq<ListHPBindData>(element_type); 
    }
    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<ListHPBindData>();
        return element_type == other.element_type;
    }
};

/**
 * This function determines if the given parameters are valid and selects the return type
 */
static unique_ptr<FunctionData> ListHPBind(ClientContext &, ScalarFunction &bound_function, vector<unique_ptr<Expression>> &arguments) {
    D_ASSERT(arguments.size() == 1);
    LogicalType element_type;
    LogicalType arg_type = arguments[0]->return_type;
    if (arg_type.id() != LogicalTypeId::LIST) {
        throw BinderException("%s is not supported for this function", arg_type);
    }
    while(arg_type.id() == LogicalTypeId::LIST) {
        arg_type = ListType::GetChildType(arg_type);
    }
    if (!arg_type.IsNumeric()) {
        throw BinderException("%s with LIST is not supported in this function", arg_type);
    }
    element_type = arg_type;
    bound_function.return_type = LogicalType::INTEGER;
    return make_uniq<ListHPBindData>(element_type);
}

/**
 * This function selects the correct function according to the parameter types
 */
static void ListHPExec(DataChunk &args, ExpressionState &state, Vector &result) {
    auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
    auto &info = func_expr.bind_info->Cast<ListHPBindData>();
    switch (info.element_type.id()) {
    case LogicalTypeId::FLOAT:
        HighestPositionFun<float>(args, state, result); 
        break;
    case LogicalTypeId::DOUBLE:
        HighestPositionFun<double>(args, state, result);
        break;
    case LogicalTypeId::BFLOAT:
        HighestPositionFun<std::bfloat16_t>(args, state, result);
        break;
    default:
        throw NotImplementedException("Unsupported element type for highest position");
    }
}

/**
 * Registers highest position function
 */
ScalarFunctionSet HighestPosition::GetFunctions() {
	ScalarFunctionSet set("highestposition");
	set.AddFunction(ScalarFunction({LogicalType::ANY}, LogicalType::ANY, ListHPExec, ListHPBind));
	for (auto &func : set.functions) {
		BaseScalarFunction::SetReturnsError(func);
	}
	return set;
}
} // namespace duckdb
