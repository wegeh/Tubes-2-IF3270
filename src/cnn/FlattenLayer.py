class FlattenLayer:
    def forward(self, x):
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)
