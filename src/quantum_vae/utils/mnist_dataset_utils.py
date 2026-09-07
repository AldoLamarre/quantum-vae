from .script_utils import LabelMappedDataset, filter_dataset_by_labels, split_train_val


def build_label_mapped_splits(
    training_data,
    test_data,
    label_map,
    train_ratio=0.9,
):
    filtered_train = LabelMappedDataset(filter_dataset_by_labels(training_data, label_map), label_map)
    filtered_test = LabelMappedDataset(filter_dataset_by_labels(test_data, label_map), label_map)
    train_set, val_set = split_train_val(filtered_train, train_ratio)
    return filtered_train, filtered_test, train_set, val_set
