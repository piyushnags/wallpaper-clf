# CL Imports (avalanche-lib)
from avalanche.training.templates import SupervisedTemplate



class HFSupervised(SupervisedTemplate):
    '''
    Description:
        Integrating HuggingFace's ViT model
        with the Avalanche library
    '''
    @property
    def mb_x(self):
        '''
        Inputs of minibatch are first element
        of the mbatch list
        '''
        return self.mbatch[0]
    

    @property
    def mb_y(self):
        '''
        Target of minibatch are second element
        of the mbatch list
        '''
        return self.mbatch[1]
    

    def _unpack_minibatch(self):
        '''
        Move minibatch to device
        In this case, the minibatch is a 
        list containing all inputs (x), targets(y),
        and task labels (t) in a list containing 3 torch 
        tensors
        '''
        for item in self.mbatch:
            item = item.to(self.device)
    

    def criterion(self):
        '''
        Compute the loss
        '''
        loss = self._criterion(self.mb_output, self.mb_y)
        return loss


    def forward(self):
        '''
        forward impl for the train/eval call
        '''
        inputs = {
            'pixel_values' : self.mb_x,
            'labels' : self.mb_y
        }
        output = self.model(**inputs)
        return output.logits



if __name__ == '__main__':
    pass