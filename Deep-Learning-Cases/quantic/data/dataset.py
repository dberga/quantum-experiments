
class Dataset():
    """ Generic dataset class

    Provides a common functionality among datasets

    Contains the following general dataset information:
        - framework (str): format data according to specific framework ('torchvision','numpy','tensorflow'...)
        - partitions (dict): dictionary containing multiple dataset partitions ('train','val','test'...)

    Args:
        framework (str): format dataset to specific framework ('torchvision','numpy'...)

    """

    def __init__(self,framework=None,*kargs,**kwargs):
        self.framework = framework
        self.partitions = {}

    def get_partitions(self):
        """ Get avilable partitions names
        
        Returns:
            list[str]: list of partition names
        """
        return list(self.partitions.keys())

    def __getitem__(self,partition_name):
        """ Get a dataset partition by specifygin its name

        Args:
            partition_name (str): name of the partition to retrieve
        
        Returns:
            : dataset partition (format depends on framework specified)
        """

        return self.partitions[partition_name]
        





