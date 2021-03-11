import torch

class DataLoader:
  def __init__(self, vocabulary, labels, stop_words, train_set):
    self.vocabulary = vocabulary
    self.labels = labels
    self.stop_words = stop_words
    self.train_set = train_set
    self.pt = 0
    self.length = len(train_set)

  def get_sent_offset(self, feature, label, labels, vocabulary, stop_words):
    # label, feature = sent
    label = torch.LongTensor([labels.index(label)])
    offset = []
    feature = feature.lower()
    for word in feature.split():
      if word in stop_words:
        continue
      if word in vocabulary:
        offset.append(vocabulary.index(word))
      else:
        offset.append(vocabulary.index('#unk#'))
    return torch.LongTensor(offset).unsqueeze(-2), label

  def get_batch(self):
    if self.pt + 1 < self.length:
      label, feat = self.train_set[self.pt:self.pt+1][0]
      feat, label = self.get_sent_offset(
                        feat,
                        label,
                        self.labels,
                        self.vocabulary,
                        self.stop_words
                      )
      self.pt += 1
      return feat, label
    else:
      self.pt = 0
      label, feat = self.train_set[self.pt:self.pt+1][0]
      feat, label = self.get_sent_offset(
                        feat,
                        label,
                        self.labels,
                        self.vocabulary,
                        self.stop_words
                      )
      self.pt += 1
      return feat, label
