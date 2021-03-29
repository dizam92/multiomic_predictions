from multiomic_modeling.models.base import Model, CustomModelCheckpoint

class MultiomicPredictionModel(Model):
    def __init__(self, encoder, decoder, loss: str = 'ce'):
        self.encoder = encoder
        self.decoder = decoder
        if loss.lower() == 'ce':
            self.__loss = torch.nn.CrossEntropyLoss()
        else:
            raise f'The error {loss} is not supported yet'
        
    def forward(self, inputs, target) -> torch.Tensor:
        enc_res = self.encoder(inputs)
        output = self.decoder(target, enc_res)
        return output
    
    def predict(self, inputs, targets=None):
        # outputs = self(inputs, targets)
        return self(inputs, targets)
            
    def attention_scores(self, inputs):
        return self.encoder(inputs).attention_scores

    def compute_loss_metrics(self, preds, targets):
        return self.__loss(preds, targets)
    