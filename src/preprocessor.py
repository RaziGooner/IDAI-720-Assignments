from collections import Counter
import numpy as np


def Reweighing(X, Y, A):
    # X: independent variables (2-d pd.DataFrame)
    # Y: the dependent variable (1-d np.array, binary y in {0,1})
    # A: a list/array of the names of the sensitive attributes with binary values
    # Return: sample_weight, an array of float weight for every data point
    #         sample_weight(a,y) = P(y)*P(a)/P(a,y)
    # Write your code below:
    A_df = X[A]

    # Calculate joint probabilities P(a,y)
    ay_counts = Counter([(a_val, y_val) for a_val, y_val in zip(A_df.itertuples(index=False, name=None), Y)])
    P_ay = {k: v / len(Y) for k, v in ay_counts.items()}

    # Calculate marginal probabilities P(a) and P(y)
    a_counts = A_df.apply(lambda x: tuple(x), axis=1).value_counts(normalize=True)
    y_counts = Counter(Y)
    P_a = dict(a_counts)
    P_y = {k: v / len(Y) for k, v in y_counts.items()}

    # Calculate sample weights
    sample_weight = np.array(
        [P_y[y_val] * P_a[tuple(a_val)] / P_ay[(tuple(a_val), y_val)] for a_val, y_val in zip(A_df.to_numpy(), Y)])

    # Rescale the sum of sample weights to len(y) before returning it
    sample_weight = sample_weight * len(Y) / sum(sample_weight)
    return sample_weight
