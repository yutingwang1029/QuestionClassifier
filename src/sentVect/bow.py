import torch

class BOW(torch.nn.Module):
  def __init__(
    self,
    vocab_size,
    embedding_dim,
    from_pretrain=False,
    pre_train_weight=None,
    freeze=False
  ):
    super(BOW, self).__init__()
    if from_pretrain == False:
      self.embeddingBag = torch.nn.EmbeddingBag(vocab_size, embedding_dim, mode='mean')
    else:
      self.embeddingBag = torch.nn.EmbeddingBag.from_pretrained(pre_train_weight, freeze=freeze, mode='mean')

  def forward(self, x):
    out = self.embeddingBag(x)
    return out