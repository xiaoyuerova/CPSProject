from torch import nn
from tcn_test_0.data_tcn.parameters import Parameters

parameters = Parameters()


class TextClassification(nn.Module):

    def __init__(self, vocab_size, num_class, embed_dim=parameters.embedding_size):
        super(TextClassification, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        init_range = 0.5
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        """
        text torch.Size([44])
        embedded torch.Size([8, 100])
        output torch.Size([8, 12])
        embeddingBag 会对每个bag进行sum操作
        """
        embedded = self.embedding(text, offsets)
        output = self.fc(embedded)
        return output
