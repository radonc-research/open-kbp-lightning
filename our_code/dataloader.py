from our_code.dataset import CTDoseDataSet
from torch.utils.data import DataLoader
import math

class CTDoseDataLoader():
    def __init__(self, path, batch_size=1,
                 patient_shape=(128, 128, 128),
                 shuffle=True,
                 mode_name='training_model',
                 trans=['flip', 64]):

        self.path = path
        self.batch_size = batch_size
        self.patient_shape = patient_shape
        self.shuffle = shuffle
        self.mode_name = mode_name

        self.dataset = CTDoseDataSet(path, patient_shape=self.patient_shape,
                                     mode_name=self.mode_name, trans=trans)

        self.dataloader = DataLoader(self.dataset,
                                       batch_size=self.batch_size,
                                       shuffle=self.shuffle,
                                       num_workers=2)


    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data

    def __len__(self):
        return math.ceil(len(self.dataset)/self.batch_size)
