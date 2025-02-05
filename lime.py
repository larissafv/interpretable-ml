def get_slice_width(ecg, num_slices=30):
    slice_width = ecg.shape[1] // num_slices
    return slice_width

def plot_segmented_ecg(ecg, slice_width, num_slices, label_leads = ["DI", "DII", "DIII", "AVL", "AVF", "AVR", "V1", "V2", "V3", "V4", "V5", "V6"]):
    
    fig = plt.figure(figsize=(12, 45))
    ax1 = fig.add_subplot(12, 1, 1)
    ax2 = fig.add_subplot(12, 1, 2, sharex=ax1)
    ax3 = fig.add_subplot(12, 1, 3, sharex=ax1)
    ax4 = fig.add_subplot(12, 1, 4, sharex=ax1)
    ax5 = fig.add_subplot(12, 1, 5, sharex=ax1)
    ax6 = fig.add_subplot(12, 1, 6, sharex=ax1)
    ax7 = fig.add_subplot(12, 1, 7, sharex=ax1)
    ax8 = fig.add_subplot(12, 1, 8, sharex=ax1)
    ax9 = fig.add_subplot(12, 1, 9, sharex=ax1)
    ax10 = fig.add_subplot(12, 1, 10, sharex=ax1)
    ax11 = fig.add_subplot(12, 1, 11, sharex=ax1)
    ax12 = fig.add_subplot(12, 1, 12, sharex=ax1)

    ax1.plot(ecg[0], c='b')
    ax2.plot(ecg[1], c='b')
    ax3.plot(ecg[2], c='b')
    ax4.plot(ecg[3], c='b')
    ax5.plot(ecg[4], c='b')
    ax6.plot(ecg[5], c='b')
    ax7.plot(ecg[6], c='b')
    ax8.plot(ecg[7], c='b')
    ax9.plot(ecg[8], c='b')
    ax10.plot(ecg[9], c='b')
    ax11.plot(ecg[10], c='b')
    ax12.plot(ecg[11], c='b')

    for i in range(1, num_slices):
       ax1.axvline(x=i * slice_width, color='r', linestyle='--')
       ax2.axvline(x=i * slice_width, color='r', linestyle='--')
       ax3.axvline(x=i * slice_width, color='r', linestyle='--')
       ax4.axvline(x=i * slice_width, color='r', linestyle='--')
       ax5.axvline(x=i * slice_width, color='r', linestyle='--')
       ax6.axvline(x=i * slice_width, color='r', linestyle='--')
       ax7.axvline(x=i * slice_width, color='r', linestyle='--')
       ax8.axvline(x=i * slice_width, color='r', linestyle='--')
       ax9.axvline(x=i * slice_width, color='r', linestyle='--')
       ax10.axvline(x=i * slice_width, color='r', linestyle='--')
       ax11.axvline(x=i * slice_width, color='r', linestyle='--')
       ax12.axvline(x=i * slice_width, color='r', linestyle='--')

    ax1.set_title(label_leads[0])
    ax2.set_title(label_leads[1])
    ax3.set_title(label_leads[2])
    ax4.set_title(label_leads[3])
    ax5.set_title(label_leads[4])
    ax6.set_title(label_leads[5])
    ax7.set_title(label_leads[6])
    ax8.set_title(label_leads[7])
    ax9.set_title(label_leads[8])
    ax10.set_title(label_leads[9])
    ax11.set_title(label_leads[10])
    ax12.set_title(label_leads[11])

    plt.show()

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

def plot_perturbed_ecg(original_ecg, perturbed_ecg, perturbation, num_slices, slice_width, label_leads = ["DI", "DII", "DIII", "AVL", "AVF", "AVR", "V1", "V2", "V3", "V4", "V5", "V6"]):
    total_length = len(original_ecg[0])

    fig = plt.figure(figsize=(25, 45))
    ax1 = fig.add_subplot(12, 2, 1)
    ax1_ = fig.add_subplot(12, 2, 2)
    ax2 = fig.add_subplot(12, 2, 3, sharex=ax1)
    ax2_ = fig.add_subplot(12, 2, 4, sharex=ax1_)
    ax3 = fig.add_subplot(12, 2, 5, sharex=ax1)
    ax3_ = fig.add_subplot(12, 2, 6, sharex=ax1_)
    ax4 = fig.add_subplot(12, 2, 7, sharex=ax1)
    ax4_ = fig.add_subplot(12, 2, 8, sharex=ax1_)
    ax5 = fig.add_subplot(12, 2, 9, sharex=ax1)
    ax5_ = fig.add_subplot(12, 2, 10, sharex=ax1_)
    ax6 = fig.add_subplot(12, 2, 11, sharex=ax1)
    ax6_ = fig.add_subplot(12, 2, 12, sharex=ax1_)
    ax7 = fig.add_subplot(12, 2, 13, sharex=ax1)
    ax7_ = fig.add_subplot(12, 2, 14, sharex=ax1_)
    ax8 = fig.add_subplot(12, 2, 15, sharex=ax1)
    ax8_ = fig.add_subplot(12, 2, 16, sharex=ax1_)
    ax9 = fig.add_subplot(12, 2, 17, sharex=ax1)
    ax9_ = fig.add_subplot(12, 2, 18, sharex=ax1_)
    ax10 = fig.add_subplot(12, 2, 19, sharex=ax1)
    ax10_ = fig.add_subplot(12, 2, 20, sharex=ax1_)
    ax11 = fig.add_subplot(12, 2, 21, sharex=ax1)
    ax11_ = fig.add_subplot(12, 2, 22, sharex=ax1_)
    ax12 = fig.add_subplot(12, 2, 23, sharex=ax1)
    ax12_ = fig.add_subplot(12, 2, 24, sharex=ax1_)

    ax1.plot(original_ecg[0], c='black')
    ax2.plot(original_ecg[1], c='black')
    ax3.plot(original_ecg[2], c='black')
    ax4.plot(original_ecg[3], c='black')
    ax5.plot(original_ecg[4], c='black')
    ax6.plot(original_ecg[5], c='black')
    ax7.plot(original_ecg[6], c='black')
    ax8.plot(original_ecg[7], c='black')
    ax9.plot(original_ecg[8], c='black')
    ax10.plot(original_ecg[9], c='black')
    ax11.plot(original_ecg[10], c='black')
    ax12.plot(original_ecg[11], c='black')

    for i in range(num_slices):
       start_idx = i * slice_width
       end_idx = min((i + 1) * slice_width, total_length)

       ax1.axvline(x=start_idx, color='r', linestyle='--')
       if perturbation[0][i] == 0:
          ax1.axvspan(start_idx, end_idx, color='red', alpha=0.3)
       ax2.axvline(x=start_idx, color='r', linestyle='--')
       if perturbation[1][i] == 0:
          ax2.axvspan(start_idx, end_idx, color='red', alpha=0.3)
       ax3.axvline(x=start_idx, color='r', linestyle='--')
       if perturbation[2][i] == 0:
          ax3.axvspan(start_idx, end_idx, color='red', alpha=0.3)
       ax4.axvline(x=start_idx, color='r', linestyle='--')
       if perturbation[3][i] == 0:
          ax4.axvspan(start_idx, end_idx, color='red', alpha=0.3)
       ax5.axvline(x=start_idx, color='r', linestyle='--')
       if perturbation[4][i] == 0:
          ax5.axvspan(start_idx, end_idx, color='red', alpha=0.3)
       ax6.axvline(x=start_idx, color='r', linestyle='--')
       if perturbation[5][i] == 0:
          ax6.axvspan(start_idx, end_idx, color='red', alpha=0.3)
       ax7.axvline(x=start_idx, color='r', linestyle='--')
       if perturbation[6][i] == 0:
          ax7.axvspan(start_idx, end_idx, color='red', alpha=0.3)
       ax8.axvline(x=start_idx, color='r', linestyle='--')
       if perturbation[7][i] == 0:
          ax8.axvspan(start_idx, end_idx, color='red', alpha=0.3)
       ax9.axvline(x=start_idx, color='r', linestyle='--')
       if perturbation[8][i] == 0:
          ax9.axvspan(start_idx, end_idx, color='red', alpha=0.3)
       ax10.axvline(x=start_idx, color='r', linestyle='--')
       if perturbation[9][i] == 0:
          ax10.axvspan(start_idx, end_idx, color='red', alpha=0.3)
       ax11.axvline(x=start_idx, color='r', linestyle='--')
       if perturbation[10][i] == 0:
          ax11.axvspan(start_idx, end_idx, color='red', alpha=0.3)
       ax12.axvline(x=start_idx, color='r', linestyle='--')
       if perturbation[11][i] == 0:
          ax12.axvspan(start_idx, end_idx, color='red', alpha=0.3)

    ax1_.plot(perturbed_ecg[0], c='green')
    ax2_.plot(perturbed_ecg[1], c='green')
    ax3_.plot(perturbed_ecg[2], c='green')
    ax4_.plot(perturbed_ecg[3], c='green')
    ax5_.plot(perturbed_ecg[4], c='green')
    ax6_.plot(perturbed_ecg[5], c='green')
    ax7_.plot(perturbed_ecg[6], c='green')
    ax8_.plot(perturbed_ecg[7], c='green')
    ax9_.plot(perturbed_ecg[8], c='green')
    ax10_.plot(perturbed_ecg[9], c='green')
    ax11_.plot(perturbed_ecg[10], c='green')
    ax12_.plot(perturbed_ecg[11], c='green')

    for i in range(num_slices):
       start_idx = i * slice_width
       end_idx = min((i + 1) * slice_width, total_length)

       ax1_.axvline(x=start_idx, color='r', linestyle='--')
       if perturbation[0][i] == 0:
          ax1_.axvspan(start_idx, end_idx, color='red', alpha=0.3)
       ax2_.axvline(x=start_idx, color='r', linestyle='--')
       if perturbation[1][i] == 0:
          ax2_.axvspan(start_idx, end_idx, color='red', alpha=0.3)
       ax3_.axvline(x=start_idx, color='r', linestyle='--')
       if perturbation[2][i] == 0:
          ax3_.axvspan(start_idx, end_idx, color='red', alpha=0.3)
       ax4_.axvline(x=start_idx, color='r', linestyle='--')
       if perturbation[3][i] == 0:
          ax4_.axvspan(start_idx, end_idx, color='red', alpha=0.3)
       ax5_.axvline(x=start_idx, color='r', linestyle='--')
       if perturbation[4][i] == 0:
          ax5_.axvspan(start_idx, end_idx, color='red', alpha=0.3)
       ax6_.axvline(x=start_idx, color='r', linestyle='--')
       if perturbation[5][i] == 0:
          ax6_.axvspan(start_idx, end_idx, color='red', alpha=0.3)
       ax7_.axvline(x=start_idx, color='r', linestyle='--')
       if perturbation[6][i] == 0:
          ax7_.axvspan(start_idx, end_idx, color='red', alpha=0.3)
       ax8_.axvline(x=start_idx, color='r', linestyle='--')
       if perturbation[7][i] == 0:
          ax8_.axvspan(start_idx, end_idx, color='red', alpha=0.3)
       ax9_.axvline(x=start_idx, color='r', linestyle='--')
       if perturbation[8][i] == 0:
          ax9_.axvspan(start_idx, end_idx, color='red', alpha=0.3)
       ax10_.axvline(x=start_idx, color='r', linestyle='--')
       if perturbation[9][i] == 0:
          ax10_.axvspan(start_idx, end_idx, color='red', alpha=0.3)
       ax11_.axvline(x=start_idx, color='r', linestyle='--')
       if perturbation[10][i] == 0:
          ax11_.axvspan(start_idx, end_idx, color='red', alpha=0.3)
       ax12_.axvline(x=start_idx, color='r', linestyle='--')
       if perturbation[11][i] == 0:
          ax12_.axvspan(start_idx, end_idx, color='red', alpha=0.3)

    ax1.set_title("Original "+ label_leads[0])
    ax1_.set_title("Perturbed " + label_leads[0])
    ax2.set_title("Original "+ label_leads[1])
    ax2_.set_title("Perturbed " + label_leads[1])
    ax3.set_title("Original "+ label_leads[2])
    ax3_.set_title("Perturbed " + label_leads[2])
    ax4.set_title("Original "+ label_leads[3])
    ax4_.set_title("Perturbed " + label_leads[3])
    ax5.set_title("Original "+ label_leads[4])
    ax5_.set_title("Perturbed " + label_leads[4])
    ax6.set_title("Original "+ label_leads[5])
    ax6_.set_title("Perturbed " + label_leads[5])
    ax7.set_title("Original "+ label_leads[6])
    ax7_.set_title("Perturbed " + label_leads[6])
    ax8.set_title("Original "+ label_leads[7])
    ax8_.set_title("Perturbed " + label_leads[7])
    ax9.set_title("Original "+ label_leads[8])
    ax9_.set_title("Perturbed " + label_leads[8])
    ax10.set_title("Original "+ label_leads[9])
    ax10_.set_title("Perturbed " + label_leads[9])
    ax11.set_title("Original "+ label_leads[10])
    ax11_.set_title("Perturbed " + label_leads[10])
    ax12.set_title("Original "+ label_leads[11])
    ax12_.set_title("Perturbed " + label_leads[11])

    plt.show()

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

def visualize_lime_explanation(ecg, top_influential_segments, num_slices, diagnosis, perturb_function=perturb_mean, label_leads = ["DI", "DII", "DIII", "AVL", "AVF", "AVR", "V1", "V2", "V3", "V4", "V5", "V6"]):

    fig = plt.figure(figsize=(12, 45))
    ax1 = fig.add_subplot(12, 1, 1)
    ax2 = fig.add_subplot(12, 1, 2, sharex=ax1)
    ax3 = fig.add_subplot(12, 1, 3, sharex=ax1)
    ax4 = fig.add_subplot(12, 1, 4, sharex=ax1)
    ax5 = fig.add_subplot(12, 1, 5, sharex=ax1)
    ax6 = fig.add_subplot(12, 1, 6, sharex=ax1)
    ax7 = fig.add_subplot(12, 1, 7, sharex=ax1)
    ax8 = fig.add_subplot(12, 1, 8, sharex=ax1)
    ax9 = fig.add_subplot(12, 1, 9, sharex=ax1)
    ax10 = fig.add_subplot(12, 1, 10, sharex=ax1)
    ax11 = fig.add_subplot(12, 1, 11, sharex=ax1)
    ax12 = fig.add_subplot(12, 1, 12, sharex=ax1)

    ax1.plot(ecg[0], c='black')
    ax2.plot(ecg[1], c='black')
    ax3.plot(ecg[2], c='black')
    ax4.plot(ecg[3], c='black')
    ax5.plot(ecg[4], c='black')
    ax6.plot(ecg[5], c='black')
    ax7.plot(ecg[6], c='black')
    ax8.plot(ecg[7], c='black')
    ax9.plot(ecg[8], c='black')
    ax10.plot(ecg[9], c='black')
    ax11.plot(ecg[10], c='black')
    ax12.plot(ecg[11], c='black')

    for i in range(1, num_slices):
       ax1.axvline(x=i * (len(ecg[0]) // num_slices), color='r', linestyle='--')
       ax2.axvline(x=i * (len(ecg[0]) // num_slices), color='r', linestyle='--')
       ax3.axvline(x=i * (len(ecg[0]) // num_slices), color='r', linestyle='--')
       ax4.axvline(x=i * (len(ecg[0]) // num_slices), color='r', linestyle='--')
       ax5.axvline(x=i * (len(ecg[0]) // num_slices), color='r', linestyle='--')
       ax6.axvline(x=i * (len(ecg[0]) // num_slices), color='r', linestyle='--')
       ax7.axvline(x=i * (len(ecg[0]) // num_slices), color='r', linestyle='--')
       ax8.axvline(x=i * (len(ecg[0]) // num_slices), color='r', linestyle='--')
       ax9.axvline(x=i * (len(ecg[0]) // num_slices), color='r', linestyle='--')
       ax10.axvline(x=i * (len(ecg[0]) // num_slices), color='r', linestyle='--')
       ax11.axvline(x=i * (len(ecg[0]) // num_slices), color='r', linestyle='--')
       ax12.axvline(x=i * (len(ecg[0]) // num_slices), color='r', linestyle='--')


    for segment in top_influential_segments[0]:
        start_idx = segment * (len(ecg[0]) // num_slices)
        end_idx = start_idx + (len(ecg[0]) // num_slices)
        ax1.axvspan(start_idx, end_idx, color='green', alpha=0.2)
    for segment in top_influential_segments[1]:
        start_idx = segment * (len(ecg[0]) // num_slices)
        end_idx = start_idx + (len(ecg[0]) // num_slices)
        ax2.axvspan(start_idx, end_idx, color='green', alpha=0.2)
    for segment in top_influential_segments[2]:
        start_idx = segment * (len(ecg[0]) // num_slices)
        end_idx = start_idx + (len(ecg[0]) // num_slices)
        ax3.axvspan(start_idx, end_idx, color='green', alpha=0.2)
    for segment in top_influential_segments[3]:
        start_idx = segment * (len(ecg[0]) // num_slices)
        end_idx = start_idx + (len(ecg[0]) // num_slices)
        ax4.axvspan(start_idx, end_idx, color='green', alpha=0.2)
    for segment in top_influential_segments[4]:
        start_idx = segment * (len(ecg[0]) // num_slices)
        end_idx = start_idx + (len(ecg[0]) // num_slices)
        ax5.axvspan(start_idx, end_idx, color='green', alpha=0.2)
    for segment in top_influential_segments[5]:
        start_idx = segment * (len(ecg[0]) // num_slices)
        end_idx = start_idx + (len(ecg[0]) // num_slices)
        ax6.axvspan(start_idx, end_idx, color='green', alpha=0.2)
    for segment in top_influential_segments[6]:
        start_idx = segment * (len(ecg[0]) // num_slices)
        end_idx = start_idx + (len(ecg[0]) // num_slices)
        ax7.axvspan(start_idx, end_idx, color='green', alpha=0.2)
    for segment in top_influential_segments[7]:
        start_idx = segment * (len(ecg[0]) // num_slices)
        end_idx = start_idx + (len(ecg[0]) // num_slices)
        ax8.axvspan(start_idx, end_idx, color='green', alpha=0.2)
    for segment in top_influential_segments[8]:
        start_idx = segment * (len(ecg[0]) // num_slices)
        end_idx = start_idx + (len(ecg[0]) // num_slices)
        ax9.axvspan(start_idx, end_idx, color='green', alpha=0.2)
    for segment in top_influential_segments[9]:
        start_idx = segment * (len(ecg[0]) // num_slices)
        end_idx = start_idx + (len(ecg[0]) // num_slices)
        ax10.axvspan(start_idx, end_idx, color='green', alpha=0.2)
    for segment in top_influential_segments[10]:
        start_idx = segment * (len(ecg[0]) // num_slices)
        end_idx = start_idx + (len(ecg[0]) // num_slices)
        ax11.axvspan(start_idx, end_idx, color='green', alpha=0.2)
    for segment in top_influential_segments[11]:
        start_idx = segment * (len(ecg[0]) // num_slices)
        end_idx = start_idx + (len(ecg[0]) // num_slices)
        ax12.axvspan(start_idx, end_idx, color='green', alpha=0.2)


    ax1.set_title("Highlighted Key Segments "+ label_leads[0] + ", " + diagnosis)
    ax2.set_title("Highlighted Key Segments "+ label_leads[1] + ", " + diagnosis)
    ax3.set_title("Highlighted Key Segments "+ label_leads[2] + ", " + diagnosis)
    ax4.set_title("Highlighted Key Segments "+ label_leads[3] + ", " + diagnosis)
    ax5.set_title("Highlighted Key Segments "+ label_leads[4] + ", " + diagnosis)
    ax6.set_title("Highlighted Key Segments "+ label_leads[5] + ", " + diagnosis)
    ax7.set_title("Highlighted Key Segments "+ label_leads[6] + ", " + diagnosis)
    ax8.set_title("Highlighted Key Segments "+ label_leads[7] + ", " + diagnosis)
    ax9.set_title("Highlighted Key Segments "+ label_leads[8] + ", " + diagnosis)
    ax10.set_title("Highlighted Key Segments "+ label_leads[9] + ", " + diagnosis)
    ax11.set_title("Highlighted Key Segments "+ label_leads[10] + ", " + diagnosis)
    ax12.set_title("Original "+ label_leads[11] + ", " + diagnosis)

    plt.show()