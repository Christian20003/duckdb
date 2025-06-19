#include "core_functions/aggregate/nested_functions.hpp"
#include "duckdb/common/types/list_segment.hpp"
#include <stdfloat>

namespace duckdb {

/**
 * Function that will generate the resulting sum
 */
template<class TYPE>
static void SumOperation(const TYPE *input, TYPE *result, idx_t input_size, idx_t result_size) {
	for(idx_t i = 0; i < input_size; i += result_size) {
		for(idx_t j = 0; j < result_size; j++) {
			if (i == 0) {
				*(result + j) = *(input + j);
			} else {
				*(result + j) += *(input + i + j);
			}
		}
	}
}

/**
 * This struct stores important properties for manipulating the sum state
 */
struct ListBindSumData : public FunctionData {
	explicit ListBindSumData(const LogicalType &return_type, const LogicalType &element_type);
	~ListBindSumData() override;

	LogicalType return_type;
	LogicalType element_type;
	ListSegmentFunctions functions;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<ListBindSumData>(return_type, element_type);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<ListBindSumData>();
		return return_type == other.return_type && element_type == other.element_type;
	}
};

ListBindSumData::ListBindSumData(const LogicalType &return_type, const LogicalType &element_type) : return_type(return_type), element_type(element_type) {
	GetSegmentDataFunctions(functions, return_type);
}

ListBindSumData::~ListBindSumData() {
}

/**
 * This struct stores the sum state
 */
struct ListSumState {
	LinkedList linked_list;
	idx_t result_length;
};

struct ListSumFunction {
	template <class STATE>
	static void Initialize(STATE &state) {
		state.linked_list.total_capacity = 0;
		state.linked_list.first_segment = nullptr;
		state.linked_list.last_segment = nullptr;
		state.result_length = 0;
	}
	static bool IgnoreNull() {
		return false;
	}
};

/**
 * This function will be executed on each row
 * (Each row will get a sum state)
 */
static void ListSumUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                               Vector &state_vector, idx_t count) {

	D_ASSERT(input_count == 1);
	auto &input = inputs[0];
	RecursiveUnifiedVectorFormat input_data;
	Vector::RecursiveToUnifiedFormat(input, count, input_data);

	UnifiedVectorFormat states_data;
	state_vector.ToUnifiedFormat(count, states_data);
	auto states = UnifiedVectorFormat::GetData<ListSumState *>(states_data);

	auto &list_bind_data = aggr_input_data.bind_data->Cast<ListBindSumData>();

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[states_data.sel->get_index(i)];
		aggr_input_data.allocator.AlignNext();
		list_bind_data.functions.AppendRow(aggr_input_data.allocator, state.linked_list, input_data, i);
	}

}

/**
 * This function merges each row state into a single result state
 */
static void ListSumCombine(Vector &states_vector, Vector &combined, AggregateInputData &aggr_input_data,
                                idx_t count) {

	UnifiedVectorFormat states_data;
	states_vector.ToUnifiedFormat(count, states_data);
	auto states_ptr = UnifiedVectorFormat::GetData<const ListSumState *>(states_data);
	auto combined_ptr = FlatVector::GetData<ListSumState *>(combined);

	auto &list_bind_data = aggr_input_data.bind_data->Cast<ListBindSumData>();
	auto result_type = list_bind_data.return_type;

	auto &source = *states_ptr[states_data.sel->get_index(0)];
	auto &target = *combined_ptr[0];
	auto capacity = source.linked_list.total_capacity;

	// Write states of each row into the defined vector
	Vector input(result_type, capacity);
	list_bind_data.functions.BuildListVector(source.linked_list, input, 0);

	// Change the format of the new vector
	RecursiveUnifiedVectorFormat input_data;
	Vector::RecursiveToUnifiedFormat(input, capacity, input_data);

	// Vector which stores the sum values
	Vector final_state(result_type);
	// Number of elements (include multiple rows)
	auto length = capacity;
	// Number of elements (include single row)
	idx_t size = 1;
	// Pointer to the last created vector
	auto *state_vec = &final_state;
	// Pointer to the input vector and children to get access of individual sizes
	auto *data_list_vec = &input;
	// Pointer to the input vector and children in unfified-vector-format
	auto *data_uniform_vec = &input_data;
	// Iterate over all children of type LIST
	while(data_uniform_vec->logical_type.id() == LogicalTypeId::LIST) {
		// Get access to child lengths
		auto metadata = UnifiedVectorFormat::GetData<list_entry_t>(data_uniform_vec->unified);
		idx_t dimension_val = 0;
		for(idx_t i = 0; i < length; i++) {
			// Set current dimension size
			if (i == 0) {
				dimension_val = metadata[i].length;
			// Proof if consistent over all entries
			} else if (metadata[i].length != dimension_val) {
				throw InvalidInputException("Lists do not have the same size");
			}
		}
		// Specify metadata of result Vector
		if (size == 1) {
			ListVector::Reserve(final_state, dimension_val);
			auto *metadata = ListVector::GetData(final_state);
			metadata->offset = 0;
			metadata->length = dimension_val;
			target.result_length = dimension_val;
		} else {
			state_vec = &ListVector::GetEntry(*state_vec);
		}
		size = size * dimension_val;
		length = ListVector::GetListSize(*data_list_vec);
		data_list_vec = &ListVector::GetEntry(*data_list_vec);
		data_uniform_vec = &data_uniform_vec->children.back();
	}

	// Copy the first element of the input to the final_state
	VectorOperations::Copy(input, final_state, 1, 0, 0);

	// Get the actual data of input and result
	auto data_vec = ListVector::GetEntry(*state_vec);

	// Execute sum operation based on the element type
	if (list_bind_data.element_type.id() == LogicalTypeId::FLOAT) {
		auto *result_data = FlatVector::GetData<float>(data_vec);
		auto *incoming_data = UnifiedVectorFormat::GetData<float>(data_uniform_vec->unified);
		SumOperation<float>(incoming_data, result_data, size * capacity, size);		
	} else if (list_bind_data.element_type.id() == LogicalTypeId::DOUBLE) {
		auto *result_data = FlatVector::GetData<double>(data_vec);
		auto *incoming_data = UnifiedVectorFormat::GetData<double>(data_uniform_vec->unified);
		SumOperation<double>(incoming_data, result_data, size * capacity, size);
	} else if (list_bind_data.element_type.id() == LogicalTypeId::BFLOAT) {
		auto *result_data = FlatVector::GetData<std::bfloat16_t>(data_vec);
		auto *incoming_data = UnifiedVectorFormat::GetData<std::bfloat16_t>(data_uniform_vec->unified);
		SumOperation<std::bfloat16_t>(incoming_data, result_data, size * capacity, size);
	} else {
		throw InvalidInputException("Type %s is not supported", list_bind_data.element_type);
	}

	// Transform final vector into unified-vector-format
	RecursiveUnifiedVectorFormat final_data;
	Vector::RecursiveToUnifiedFormat(final_state, 1, final_data);

	// Write content of result vector into the state object.
	idx_t entry_idx = 0;
	aggr_input_data.allocator.AlignNext();
	list_bind_data.functions.AppendRow(aggr_input_data.allocator, target.linked_list, final_data, entry_idx);
}

/*
 * This function will generate the result vector
 */
static void ListSumFinalize(Vector &states_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                         idx_t offset) {

	UnifiedVectorFormat states_data;
	states_vector.ToUnifiedFormat(count, states_data);
	auto states = UnifiedVectorFormat::GetData<ListSumState *>(states_data);

	D_ASSERT(result.GetType().id() == LogicalTypeId::LIST);

	auto &mask = FlatVector::Validity(result);
	auto result_data = FlatVector::GetData<list_entry_t>(result);
	size_t total_len = ListVector::GetListSize(result);

	auto &list_bind_data = aggr_input_data.bind_data->Cast<ListBindSumData>();

	// Get the resulting state
	auto &state = *states[states_data.sel->get_index(0)];
	result_data->offset = total_len;
	if (state.linked_list.total_capacity == 0) {
		mask.SetInvalid(0);
		result_data->length = 0;
		return;
	}

	// set the length and offset of this list in the result vector
	result_data->length = state.result_length;
	total_len += state.result_length;

	// reserve capacity, then iterate over all entries again and copy over the data to the child vector
	ListVector::Reserve(result, total_len);
	list_bind_data.functions.BuildListVector(state.linked_list, result, 0);

	ListVector::SetListSize(result, total_len);
}

/**
 * This function will determine the return type and checks if the input type is valid
 */
unique_ptr<FunctionData> ListSumBindFunction(ClientContext &context, AggregateFunction &function,
                                          vector<unique_ptr<Expression>> &arguments) {
	D_ASSERT(arguments.size() == 1);
	D_ASSERT(function.arguments.size() == 1);

	if (arguments[0]->return_type.id() == LogicalTypeId::UNKNOWN) {
		function.arguments[0] = LogicalTypeId::UNKNOWN;
		function.return_type = LogicalType::SQLNULL;
		return nullptr;
	}

	if (arguments[0]->return_type.id() != LogicalTypeId::LIST) {
		throw BinderException("Parameter must be of type LIST");
	}
	
	LogicalType element_type = arguments[0]->return_type;
	while(element_type.id() == LogicalTypeId::LIST){
		element_type = ListType::GetChildType(element_type);
	}

	function.return_type = arguments[0]->return_type;
	return make_uniq<ListBindSumData>(function.return_type, element_type);
}

AggregateFunctionSet ListSum::GetFunctions() {
    AggregateFunctionSet result("list_sum");

	AggregateFunction fun({LogicalType::ANY}, LogicalType::ANY, AggregateFunction::StateSize<ListSumState>,
        	                AggregateFunction::StateInitialize<ListSumState, ListSumFunction>, ListSumUpdate,
        	                ListSumCombine, ListSumFinalize, nullptr, ListSumBindFunction, nullptr, nullptr, nullptr);
    result.AddFunction(fun);
    return result;
}

}