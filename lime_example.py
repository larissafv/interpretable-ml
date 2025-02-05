ecg = np.transpose(ecgs['tracings'][67])
num_slices = 60
slice_width = get_segment_info(ecg, num_slices)
plot_segmented_ecg(ecg, slice_width, num_slices)

num_perturbations = 40
perturbations = generate_random_perturbations(num_perturbations, num_slices)

perturbations_example = []
for p in perturbations:
  perturbations_example.append(p[0])
  
perturbed_ecg_example = apply_perturbation_to_ecg(ecg, np.array(perturbations_example), num_slices, slice_width)
plot_perturbed_ecg(ecg, perturbed_ecg_example, np.array(perturbations_example), num_slices, slice_width)

model_input = prepare_ecgs(ecg, perturbations, num_slices, slice_width)

lime_y_score = model.predict(model_input, batch_size=5, verbose=1)
cosine_distances = calculate_cosine_distances(perturbations, num_slices, 12)
weights = calculate_weights_from_distances(cosine_distances)
segment_importance_coefficients = fit_explainable_model(lime_y_score, perturbations, weights)
number_of_top_features = 72
top_influential_segments = identify_top_influential_segments(segment_importance_coefficients, number_of_top_features)
top_influential_segments = separe_leads_top_influential_segments(top_influential_segments)

visualize_lime_explanation(ecg, top_influential_segments[0], num_slices, "1dAVb", perturb_function=perturb_mean)