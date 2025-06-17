#include "core_functions/aggregate/nested_functions.hpp"
#include "duckdb/common/types/list_segment.hpp"
#include <stdfloat>

namespace duckdb {

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

struct ListBindSumData : public FunctionData {
	explicit ListBindSumData(const LogicalType &stype_p);
	~ListBindSumData() override;

	LogicalType stype;
	ListSegmentFunctions functions;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<ListBindSumData>(stype);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<ListBindSumData>();
		return stype == other.stype;
	}
};

ListBindSumData::ListBindSumData(const LogicalType &stype_p) : stype(stype_p) {
	GetSegmentDataFunctions(functions, stype_p);
}

ListBindSumData::~ListBindSumData() {
}

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

template<class TYPE>
static void ListSumCombine(Vector &states_vector, Vector &combined, AggregateInputData &aggr_input_data,
                                idx_t count) {

	UnifiedVectorFormat states_data;
	states_vector.ToUnifiedFormat(count, states_data);
	auto states_ptr = UnifiedVectorFormat::GetData<const ListSumState *>(states_data);
	auto combined_ptr = FlatVector::GetData<ListSumState *>(combined);

	auto &list_bind_data = aggr_input_data.bind_data->Cast<ListBindSumData>();
	auto result_type = list_bind_data.stype;

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
	auto type = ListType::GetChildType(result_type);
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
		// Add a child vector to the last created vector
		} else {
			Vector final_child(type);
			ListVector::Reserve(final_child, size * dimension_val);
			ListVector::SetListSize(final_child, size * dimension_val);
			auto* metadata = ListVector::GetData(final_child);
			for(idx_t i = 0; i < size; i++) {
				metadata[i].offset = i * dimension_val;
				metadata[i].length = dimension_val;
			}
			// Idea: final_state.append(child_1) -> child_1.append(child_2) -> ...
			// Depending on the overall dimensions of the LIST
			ListVector::Append(*state_vec, final_child, size);
			state_vec = &ListVector::GetEntry(*state_vec);
		}
		size = size * dimension_val;
		length = ListVector::GetListSize(*data_list_vec);
		data_list_vec = &ListVector::GetEntry(*data_list_vec);
		data_uniform_vec = &data_uniform_vec->children.back();
	}

	// Get the actual data of input and result
	auto data_vec = ListVector::GetEntry(*state_vec);
	auto *result_data = FlatVector::GetData<TYPE>(data_vec);
	auto *incoming_data = UnifiedVectorFormat::GetData<TYPE>(data_uniform_vec->unified);
	
	// Execute sum operation
	SumOperation<TYPE>(incoming_data, result_data, size * capacity, size);

	// Transform final vector into unified-vector-format
	RecursiveUnifiedVectorFormat final_data;
	Vector::RecursiveToUnifiedFormat(final_state, 1, final_data);

	// Write content of result vector into the state object.
	idx_t entry_idx = 0;
	aggr_input_data.allocator.AlignNext();
	list_bind_data.functions.AppendRow(aggr_input_data.allocator, target.linked_list, final_data, entry_idx);
}

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

unique_ptr<FunctionData> ListSumBindFunction(ClientContext &context, AggregateFunction &function,
                                          vector<unique_ptr<Expression>> &arguments) {
	D_ASSERT(arguments.size() == 1);
	D_ASSERT(function.arguments.size() == 1);

	if (arguments[0]->return_type.id() == LogicalTypeId::UNKNOWN) {
		function.arguments[0] = LogicalTypeId::UNKNOWN;
		function.return_type = LogicalType::SQLNULL;
		return nullptr;
	}

	function.return_type = arguments[0]->return_type;
	return make_uniq<ListBindSumData>(function.return_type);
}

AggregateFunctionSet ListSum::GetFunctions() {
    AggregateFunctionSet result("list_sum");
    
    for (auto &type : LogicalType::Real()) {
		auto single_list = LogicalType::LIST(type);
        auto double_list = LogicalType::LIST(LogicalType::LIST(type));

		if (type.id() == LogicalTypeId::FLOAT) {
			result.AddFunction(
        	    AggregateFunction({single_list}, single_list, AggregateFunction::StateSize<ListSumState>,
        	                  AggregateFunction::StateInitialize<ListSumState, ListSumFunction>, ListSumUpdate,
        	                  ListSumCombine<float>, ListSumFinalize, nullptr, ListSumBindFunction, nullptr, nullptr, nullptr)
        	);
        	result.AddFunction(
        	    AggregateFunction({double_list}, double_list, AggregateFunction::StateSize<ListSumState>,
        	                  AggregateFunction::StateInitialize<ListSumState, ListSumFunction>, ListSumUpdate,
        	                  ListSumCombine<float>, ListSumFinalize, nullptr, ListSumBindFunction, nullptr, nullptr, nullptr)
        	);
		} else if (type.id() == LogicalTypeId::BFLOAT) {
			result.AddFunction(
        	    AggregateFunction({single_list}, single_list, AggregateFunction::StateSize<ListSumState>,
        	                  AggregateFunction::StateInitialize<ListSumState, ListSumFunction>, ListSumUpdate,
        	                  ListSumCombine<std::bfloat16_t>, ListSumFinalize, nullptr, ListSumBindFunction, nullptr, nullptr, nullptr)
        	);
        	result.AddFunction(
        	    AggregateFunction({double_list}, double_list, AggregateFunction::StateSize<ListSumState>,
        	                  AggregateFunction::StateInitialize<ListSumState, ListSumFunction>, ListSumUpdate,
        	                  ListSumCombine<std::bfloat16_t>, ListSumFinalize, nullptr, ListSumBindFunction, nullptr, nullptr, nullptr)
        	);
		} else if (type.id() == LogicalTypeId::DOUBLE) {
			result.AddFunction(
        	    AggregateFunction({single_list}, single_list, AggregateFunction::StateSize<ListSumState>,
        	                  AggregateFunction::StateInitialize<ListSumState, ListSumFunction>, ListSumUpdate,
        	                  ListSumCombine<double>, ListSumFinalize, nullptr, ListSumBindFunction, nullptr, nullptr, nullptr)
        	);
        	result.AddFunction(
        	    AggregateFunction({double_list}, double_list, AggregateFunction::StateSize<ListSumState>,
        	                  AggregateFunction::StateInitialize<ListSumState, ListSumFunction>, ListSumUpdate,
        	                  ListSumCombine<double>, ListSumFinalize, nullptr, ListSumBindFunction, nullptr, nullptr, nullptr)
        	);
		}
	}
    return result;
}

}