import json

class DatasetLoader():
    """ Loads dataset by passing the specific dataset arguments or by loading it from a coniguration JSON file"""

    _datasets = {}

    @staticmethod
    def load(dataset_type=None,from_cfg=None,*kargs,**kwargs):
        """ Load a dataset

        Args:
            dataset_type (str): dataset name to load
            from_cfg (str): load from a JSON configuration file instead
        
        Returns:
            (Dataset): dataset loaded
        """
        
        if from_cfg:
            with open(from_cfg) as f:
                cfg = json.load(f)

                if not isinstance(cfg,dict):
                    raise Exception('Invalid dataset configuration file')
                elif 'dataset_type' not in cfg:
                    raise Exception('Dataset configuration file missing \'dataset_type\' key')

            dataset_type = cfg.pop('dataset_type')
            kwargs = {**cfg, **kwargs} # Overwritte explicit configuration parameters

        elif dataset_type:
            pass
        else:
            raise Exception('Specify either \'dataset_type\' or \'from_cfg\' to load a dataset')
        

        if dataset_type not in DatasetLoader._datasets:
            raise Exception(f'Unrecognised dataset type: {dataset_type}')

        return DatasetLoader._datasets[dataset_type](*kargs,**kwargs)


    @staticmethod
    def register(name,cls_obj):
        """ Globally register a dataset type. This should be called after each Dataset class definition
        
        Args:
            name (str): name of the dataset type to register
            cls_obk(class): class module to register must be inherited from class Dataset

        """
        DatasetLoader._datasets[name] = cls_obj

