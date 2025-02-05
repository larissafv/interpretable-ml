gp = GP(
    pop_size=230,
    max_rate=0.3,
    generations=73,
    prob_crossover=0.65,
    prob_mutation=0.8,
    tournament_size=2,
    ecg=ecgs["tracings"][500],
    original_pred=y_pred[500][1],
    n_points=2700,
    n_leads=12,
    interval=250,
    model=model,
    threshold=0.09083546,
    file_name = "ecg_500_RBBB",
    label_idx = 1
)
best = gp.run()

plot_ecgs(np.transpose(ecgs['tracings'][500]), np.transpose(best))