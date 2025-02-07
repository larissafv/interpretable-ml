import numpy as np
from sklearn.linear_model import LinearRegression

def exec_global_surrogate(ecgs, y_pred):
    aux_input = np.transpose(np.array(ecgs), (0, 2, 1))
    globalsurrogate_input = []
    for ecg in aux_input:
        aux = ecg[0]
        for lead in ecg[1:]:
            aux = np.concatenate((aux, lead), axis=None)
        globalsurrogate_input.append(aux)
    globalsurrogate_input = np.array(globalsurrogate_input)
    explainable_model = LinearRegression()
    explainable_model.fit(X=globalsurrogate_input, y=y_pred)
    return explainable_model.predict(globalsurrogate_input)