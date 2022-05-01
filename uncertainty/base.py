from typing import Dict


class BaseClass(object):
    def __init__(self):
        pass

    def __call__(self, args, info: Dict) -> Dict:
        """
        Computes uncertainties based on distribution or samples
        The return should be a dictionary containing a keys pointing to name
        """
        raise NotImplementedError()
