import numpy as np
import copy
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LinearRegression

def get_slice_width(ecg, num_slices=30):
    slice_width = ecg.shape[1] // num_slices
    return slice_width

def generate_random_perturbations(num_perturbations, num_slices, n_leads = 12):
    
    leads_perturbations = []
    for lead in range(n_leads):
      leads_perturbations.append(np.random.binomial(1, 0.5, size=(num_perturbations, num_slices)))
    return np.array(leads_perturbations)

def perturb_mean(signal, start_idx, end_idx):
    mean_value = np.mean(signal[start_idx:end_idx])
    signal[start_idx:end_idx] = mean_value

def apply_perturbation_to_ecg(signal, perturbations, num_slices, slice_width, perturb_function=perturb_mean):
    # Copy the signal to avoid modifying the original
    perturbed_signal = copy.deepcopy(signal)

    for lead, lead_perturbations in enumerate(perturbations):
      for i, active in enumerate(lead_perturbations):
          start_idx = i * slice_width
          end_idx = start_idx + slice_width
          
          if not active:
              perturb_function(perturbed_signal[lead], start_idx, end_idx)

    return perturbed_signal

def prepare_ecgs(ecg, perturbations, num_slices, slice_width):
    model_input = []
    for idx_p in range(perturbations.shape[1]):
        perturbations_aux = []
        for lead_perturbations in perturbations:
            perturbations_aux.append(lead_perturbations[idx_p])
        model_input.append(apply_perturbation_to_ecg(ecg, np.array(perturbations_aux), num_slices, slice_width))

    return np.transpose(np.array(model_input), (0, 2, 1))
    
def calculate_cosine_distances(perturbations, num_slices, n_leads):
    original_ecg_rep = np.ones((1, num_slices*n_leads))

    final_perturbations = []
    for idx_p in range(perturbations.shape[1]):
        perturbations_aux = []
        for lead_perturbations in perturbations:
            perturbations_aux.append(lead_perturbations[idx_p])
        final_perturbations.append(np.ravel(np.array(perturbations_aux)))

    cosine_distances = pairwise_distances(np.array(final_perturbations), original_ecg_rep, metric='cosine').ravel()

    return cosine_distances

def calculate_weights_from_distances(cosine_distances, kernel_width=0.25):
    weights = np.sqrt(np.exp(-(cosine_distances ** 2) / kernel_width ** 2))
    return weights

def fit_explainable_model(lime_y_score, perturbations, weights):
    explainable_model = LinearRegression()

    final_perturbations = []
    for idx_p in range(perturbations.shape[1]):
        perturbations_aux = []
        for lead_perturbations in perturbations:
            perturbations_aux.append(lead_perturbations[idx_p])
        final_perturbations.append(np.ravel(np.array(perturbations_aux)))
    explainable_model.fit(X=np.array(final_perturbations), y=lime_y_score, sample_weight=weights)
    segment_importance_coefficients = explainable_model.coef_

    return segment_importance_coefficients

def identify_top_influential_segments(segment_importance_coefficients, number_of_top_features=25):
    
    top_influential_segments = []
    for target in range(segment_importance_coefficients.shape[0]):
       top_influential_segments.append(np.argsort(np.abs(segment_importance_coefficients[target]))[-number_of_top_features:])

    return np.array(top_influential_segments)

def separe_leads_top_influential_segments(top_influential_segments):
    top_influential_segments_aux = []
    for target in top_influential_segments:
        leads = [[], [], [], [], [], [], [], [], [], [], [], []]
        for s in target:
            if s < 60:
              leads[0] += [s]
            elif s < 120:
              leads[1] += [s-60]
            elif s < 180:
              leads[2] += [s-120]
            elif s < 240:
              leads[3] += [s-180]
            elif s < 300:
              leads[4] += [s-240]
            elif s < 360:
              leads[5] += [s-300]
            elif s < 420:
              leads[6] += [s-360]
            elif s < 480:
              leads[7] += [s-420]
            elif s < 540:
              leads[8] += [s-480]
            elif s < 600:
              leads[9] += [s-540]
            elif s < 660:
              leads[10] += [s-600]
            else:
              leads[11] += [s-660]
        top_influential_segments_aux.append(leads)
    return top_influential_segments_aux

def apply_final_perturbation_to_ecg(signal, perturbation, num_segments, perturb_function=perturb_mean):
    perturbed_signal = copy.deepcopy(signal)
    segment_length = len(signal) // num_segments

    for i, active in enumerate(perturbation):
        start_idx = i * segment_length
        end_idx = start_idx + segment_length
        if not active:
            perturb_function(perturbed_signal, start_idx, end_idx)

    return perturbed_signal