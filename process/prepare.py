import math
import numpy as np

class DataPreparation(object):

    def __init__(self):
        pass

    @staticmethod
    def train_val_test_split(inputs: list, outputs: list, train_s: int, val_s: int, test_s: int):

        assert np.sum([train_s, val_s, test_s]) == 100, 'Split must cover all data'

        data_count = len(inputs[0])

        train_count = math.floor(data_count * train_s/100.0)
        val_count = math.floor(data_count * val_s/100.0)
        test_count = math.floor(data_count * test_s/100.0)
        train_count+=(data_count - (train_count + val_count + test_count))

        train_val_inp, train_val_outp, test_inputs, test_outputs = DataPreparation.random_split_data(
                        inputs, outputs, data_count, train_count + val_count)

        train_inputs, train_outputs, val_inputs, val_outputs = DataPreparation.random_split_data(
                        train_val_inp, train_val_outp, train_count + val_count, train_count)
        

        print(f'Train count: {len(train_inputs[0])}\n'
              f'Val count: {len(val_inputs[0])}\n'
              f'Test count: {len(test_inputs[0])}\n'
        )
        
        return (train_inputs, train_outputs), (val_inputs, val_outputs), (test_inputs, test_outputs)


    @staticmethod
    def random_split_data(inputs: list, outputs: list, count, split_count):
        g1_ind = np.random.choice(np.arange(count), split_count, replace=False)

        mask = np.full((count), False, dtype=bool)

        mask[g1_ind] = True

        g1_inputs = []
        g2_inputs = []

        g1_outputs = []
        g2_outputs = []

        for inp in inputs:
            g1_inputs.append(inp[mask])
            g2_inputs.append(inp[~mask])

        for outp in outputs:
            g1_outputs.append(outp[mask])
            g2_outputs.append(outp[~mask])
        
        return g1_inputs, g1_outputs, g2_inputs, g2_outputs