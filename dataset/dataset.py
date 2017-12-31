'''DataSet Abstract Class
'''

class DataSet(object):
    def __init__(self, common_params, dataset_params):
        '''

        :param common_params:
        :param dataset_params:
        '''
        raise NotImplementedError

    def batch(self):
        '''

        :return:
        '''
        raise NotImplementedError

    def get_n_test_batch(self):
        '''

        :return:
        '''
        raise NotImplementedError

    def test_batch(self):
        '''

        :return:
        '''
        raise NotImplementedError

    def test_random_batch(self):
        '''

        :return:
        '''
        raise NotImplementedError