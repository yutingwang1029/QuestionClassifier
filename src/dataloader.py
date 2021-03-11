import torch

class DataLoader:
  def __init__(self, vocabulary, labels, stop_words, train_set, batch_size=1, padding=False, padding_len=25):
    self.vocabulary = vocabulary
    self.labels = labels
    self.stop_words = stop_words
    self.train_set = train_set
    self.pt = 0
    self.length = len(train_set)
    self.padding = padding
    self.padding_length = padding_len
    self.batch_size = batch_size

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
    if self.padding:
      gap = self.padding_length - len(offset)
      if gap > 0:
        for _ in range(gap):
          offset.append(vocabulary.index('#unk#'))
    return torch.LongTensor(offset).unsqueeze(-2), label

  def get_batch(self):
    if self.pt + self.batch_size < self.length:
      sents = self.train_set[self.pt:self.pt+self.batch_size]
      feats = []
      targets = []
      for sent in sents:
        label, feat = sent
        feat, label = self.get_sent_offset(
                        feat,
                        label,
                        self.labels,
                        self.vocabulary,
                        self.stop_words
                      )
        feats.append(feat.numpy().reshape(-1))
        targets.append(label)
      self.pt += self.batch_size
      return torch.LongTensor(feats), torch.LongTensor(targets)
    else:
      sents = self.train_set[self.pt:]
      feats = []
      targets = []
      for sent in sents:
        label, feat = sent
        feat, label = self.get_sent_offset(
                        feat,
                        label,
                        self.labels,
                        self.vocabulary,
                        self.stop_words
                      )
        feats.append(feat.numpy().reshape(-1))
        targets.append(label)
      self.pt = 0
      return torch.LongTensor(feats), torch.LongTensor(targets)
