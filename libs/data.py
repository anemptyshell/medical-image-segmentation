from PIL import Image
from torch.utils.data import DataLoader
import torch.utils.data

from libs.data_utils.dataset import ISIC_datasets, CHASE_datasets, ER_datasets

class data:
    def __init__(self, parser, args):
        self.parser = parser
        self.args = self.parser.get_args()

        
        
        print('#----------Preparing dataset----------#')

        if self.args.model_name == "dconnnet":

            self.train_dataset = CHASE_datasets(parser, self.args)
            self.test_dataset = CHASE_datasets(parser, self.args, train=False)

            self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, 
                                                            batch_size=self.args.batch_size, 
                                                            shuffle=True, pin_memory=True, 
                                                            num_workers=args.num_workers)
            self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, 
                                                           batch_size=1, shuffle=False, 
                                                           pin_memory=True, num_workers=args.num_workers)
            

        elif self.args.dataset == "ER":

            self.train_dataset = ER_datasets(parser, self.args)
            self.test_dataset = ER_datasets(parser, self.args, train=False)
            self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, 
                                                            batch_size=self.args.batch_size, 
                                                            shuffle=True, pin_memory=True, 
                                                            num_workers=args.num_workers)
            self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, 
                                                           batch_size=1, shuffle=False, 
                                                           pin_memory=True, num_workers=args.num_workers)


        else:
            self.train_dataset = ISIC_datasets(parser, self.args)
            self.train_loader = DataLoader(self.train_dataset,
                                       batch_size = args.batch_size, 
                                       shuffle=True,
                                       pin_memory=True,
                                       num_workers=args.num_workers,
                                    #    drop_last=True
                                       )  
        
            self.test_dataset = ISIC_datasets(parser, self.args, train=False)
            self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      pin_memory=True, 
                                      num_workers=args.num_workers,
                                    #   drop_last=True
                                      )

            # self.train_dataset = CHASE_datasets(parser, self.args)
            # self.test_dataset = CHASE_datasets(parser, self.args, train=False)

            # self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, 
            #                                                 batch_size=self.args.batch_size, 
            #                                                 shuffle=True, pin_memory=True, 
            #                                                 num_workers=args.num_workers)
            # self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, 
            #                                                batch_size=1, shuffle=False, 
            #                                                pin_memory=True, num_workers=args.num_workers)
            

        
        
    
    def get_loader(self):
        print('#----------Dataset prepared-----------#')
        return self.train_loader, self.test_loader
      
    
        


