def get_label_dict(y):
  """
  description:
    encode labels
  params:
    y: raw label list
  return:
    label2idx: dict from label to idx
    idx2label: dict from idx to label
  """
  label2idx, idx2label = {}, {}
  idx = 0
  for label in y:
    if label not in label2idx:
      label2idx[label] = idx
      idx2label[idx] = label
      idx += 1
  return label2idx, idx2label

def get_tags(sent_labels, label2idx):
  """
  description:
    convert sentence labels to idxs
  params:
    sent_labels: labels of sentences
    label2idx: dict from label to idx
  return
    list: list of idx
  """
  return [label2idx[label] for label in sent_labels]

def get_label_vecs(sent_labels, label2idx):
  tags = get_tags(sent_labels, label2idx)
  vecs = []
  for tag in tags:
    vec = [0 for _ in range(len(label2idx))]
    vec[tag] = 1
    vecs.append(vec)
  return vecs