class MyTrainer(Trainer):
    def __init__(self, loss_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_name= loss_name

    def compute_loss(self, model, inputs, return_outputs=False):
        ################################################################ 요부분!
        # config에 저장된 loss_name에 따라 다른 loss 계산 
        if self.loss_name == 'FocalLoss':
            custom_loss = FocalLoss(gamma=0.5)
                  
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs)

        if labels is not None:
            
        ################################################################ 요부분!
            loss = custom_loss(outputs[0], labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss
