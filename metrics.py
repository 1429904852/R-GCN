
# encoding=utf-8

def mean(item):
    return sum(item) / len(item)


def get_chunk_type(index, index_to_label):
    label_name = index_to_label[index]
    label_class, label_type = label_name.split("-")

    return label_name, label_class, label_type


def get_chunk(sequence, label_to_index):
    unentry = [label_to_index["O"], label_to_index["X"]]
    index_to_label = {index: label for label, index in label_to_index.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for index, label in enumerate(sequence):
        if label in unentry:
            if chunk_type is None:
                continue
            else:
                chunk = (chunk_type, chunk_start, index-1)
                chunks.append(chunk)
                chunk_type, chunk_start = None, None

        if label not in unentry:
            label_name, label_chunk_class, label_chunk_type = get_chunk_type(label, index_to_label)
            if chunk_type is None:
                chunk_type, chunk_start = label_chunk_type, index
            elif label_chunk_type == chunk_type:
                if index == (len(sequence) - 1):
                    chunk = (chunk_type, chunk_start, index)
                    chunks.append(chunk)
                elif label_chunk_class == "B":
                    chunk = (chunk_type, chunk_start, index - 1)
                    chunks.append(chunk)
                    chunk_type, chunk_start = label_chunk_type, index
                else:
                    continue
            elif label_chunk_type != chunk_type:
                chunk = (chunk_type, chunk_start, index-1)
                chunks.append(chunk)
                chunk_type, chunk_start = label_chunk_type, index

    return chunks


def gen_metrics(true_y, pred_y, label_to_index):
    correct_preds = 0
    all_preds = 0
    all_trues = 0

    true_chunks = get_chunk(true_y, label_to_index)
    pred_chunks = get_chunk(pred_y, label_to_index)

    correct_preds += len(set(true_chunks) & set(pred_chunks))
    all_preds += len(pred_chunks)
    all_trues += len(true_chunks)

    precision = correct_preds / all_preds if correct_preds > 0 else 0
    recall = correct_preds / all_trues if correct_preds > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if correct_preds > 0 else 0

    return round(f1, 4), round(precision, 4), round(recall, 4)


def gen_metrics_type(true_y, pred_y, label_to_index, label_type):
    correct_preds = 0
    all_preds = 0
    all_trues = 0
    
    true_chunks = get_chunk(true_y, label_to_index)
    pred_chunks = get_chunk(pred_y, label_to_index)

    true_class_type = []
    pre_class_type = []
    for t_chunk in true_chunks:
        if t_chunk[0] == label_type:
            true_class_type.append(t_chunk)
    
    for p_chunk in pred_chunks:
        if p_chunk[0] == label_type:
            pre_class_type.append(p_chunk)
    
    correct_preds += len(set(true_class_type) & set(pre_class_type))
    all_preds += len(pre_class_type)
    all_trues += len(true_class_type)

    precision = correct_preds / all_preds if correct_preds > 0 else 0
    recall = correct_preds / all_trues if correct_preds > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if correct_preds > 0 else 0

    return round(f1, 4), round(precision, 4), round(recall, 4)