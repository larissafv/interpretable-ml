def change_input(ecgs, lead, idx, mod):
  final_input = []
  for ecg in ecgs:
    aux = ecg.copy()
    for i in range(64):
      aux[idx+i][lead] = mod
    final_input.append(aux)
  return np.array(final_input)

def normalize_score(y_true, y_score):
  _, _, _, threshold = get_optimal_precision_recall(y_true, y_score)
  mask = y_score > threshold
  y_pred = np.zeros_like(y_score)
  y_pred[mask] = 1
  return y_pred

def permutation_feature_importance(ecgs, model, y_pred, feat_max=10.0, feat_min=-10.0):
  all_acc = []
  for intervalo in range(733, 768):
    n = intervalo * 64
    lead = int(n/4096)
    idx = n - (4096*lead)
    acc = []
    for mod in [feat_max, feat_min]:
      new_input = change_input(ecgs, lead, idx, mod)
      y_new = model.predict(new_input, batch_size=10, verbose=1)
      y_new = normalize_score(y_pred, y_new)
      acc.append(accuracy_score(y_pred, y_new))
    all_acc.append(np.mean(np.array(acc)))
  return all_acc
  
def plot_pfi_coefficients(ecg, coeff, diagnosis, label_leads = ["DI", "DII", "DIII", "AVL", "AVF", "AVR", "V1", "V2", "V3", "V4", "V5", "V6"]):
    x = np.linspace(0, len(ecg[0]))
    extent = [x[0]-(x[1]-x[0])/2., x[-1]+(x[1]-x[0])/2.,0,1]

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

    y1 = coeff[:len(ecg[0])]
    ax1.imshow(y1[np.newaxis,:], cmap="plasma", aspect="auto", extent=extent)
    ax1.set_yticks([])
    ax1.set_xlim(extent[0], extent[1])

    y2 = coeff[len(ecg[0]):2*len(ecg[0])]
    ax2.imshow(y2[np.newaxis,:], cmap="plasma", aspect="auto", extent=extent)
    ax2.set_yticks([])
    ax2.set_xlim(extent[0], extent[1])

    y3 = coeff[2*len(ecg[0]):3*len(ecg[0])]
    ax3.imshow(y3[np.newaxis,:], cmap="plasma", aspect="auto", extent=extent)
    ax3.set_yticks([])
    ax3.set_xlim(extent[0], extent[1])

    y4 = coeff[3*len(ecg[0]):4*len(ecg[0])]
    ax4.imshow(y4[np.newaxis,:], cmap="plasma", aspect="auto", extent=extent)
    ax4.set_yticks([])
    ax4.set_xlim(extent[0], extent[1])

    y5 = coeff[4*len(ecg[0]):5*len(ecg[0])]
    ax5.imshow(y5[np.newaxis,:], cmap="plasma", aspect="auto", extent=extent)
    ax5.set_yticks([])
    ax5.set_xlim(extent[0], extent[1])

    y6 = coeff[5*len(ecg[0]):6*len(ecg[0])]
    ax6.imshow(y6[np.newaxis,:], cmap="plasma", aspect="auto", extent=extent)
    ax6.set_yticks([])
    ax6.set_xlim(extent[0], extent[1])

    y7 = coeff[6*len(ecg[0]):7*len(ecg[0])]
    ax7.imshow(y7[np.newaxis,:], cmap="plasma", aspect="auto", extent=extent)
    ax7.set_yticks([])
    ax7.set_xlim(extent[0], extent[1])

    y8 = coeff[7*len(ecg[0]):8*len(ecg[0])]
    ax8.imshow(y8[np.newaxis,:], cmap="plasma", aspect="auto", extent=extent)
    ax8.set_yticks([])
    ax8.set_xlim(extent[0], extent[1])

    y9 = coeff[8*len(ecg[0]):9*len(ecg[0])]
    ax9.imshow(y9[np.newaxis,:], cmap="plasma", aspect="auto", extent=extent)
    ax9.set_yticks([])
    ax9.set_xlim(extent[0], extent[1])

    y10 = coeff[9*len(ecg[0]):10*len(ecg[0])]
    ax10.imshow(y10[np.newaxis,:], cmap="plasma", aspect="auto", extent=extent)
    ax10.set_yticks([])
    ax10.set_xlim(extent[0], extent[1])

    y11 = coeff[10*len(ecg[0]):11*len(ecg[0])]
    ax11.imshow(y11[np.newaxis,:], cmap="plasma", aspect="auto", extent=extent)
    ax11.set_yticks([])
    ax11.set_xlim(extent[0], extent[1])

    y12 = coeff[11*len(ecg[0]):]
    ax12.imshow(y12[np.newaxis,:], cmap="plasma", aspect="auto", extent=extent)
    ax12.set_yticks([])
    ax12.set_xlim(extent[0], extent[1])

    ax1_.plot(ecg[0], c='black')
    ax2_.plot(ecg[1], c='black')
    ax3_.plot(ecg[2], c='black')
    ax4_.plot(ecg[3], c='black')
    ax5_.plot(ecg[4], c='black')
    ax6_.plot(ecg[5], c='black')
    ax7_.plot(ecg[6], c='black')
    ax8_.plot(ecg[7], c='black')
    ax9_.plot(ecg[8], c='black')
    ax10_.plot(ecg[9], c='black')
    ax11_.plot(ecg[10], c='black')
    ax12_.plot(ecg[11], c='black')

    ax1.set_title("Importance Features "+ label_leads[0] + ", " + diagnosis)
    ax1_.set_title("ECG " + label_leads[0])
    ax2.set_title("Importance Features "+ label_leads[1] + ", " + diagnosis)
    ax2_.set_title("ECG " + label_leads[1])
    ax3.set_title("Importance Features "+ label_leads[2] + ", " + diagnosis)
    ax3_.set_title("ECG " + label_leads[2])
    ax4.set_title("Importance Features "+ label_leads[3] + ", " + diagnosis)
    ax4_.set_title("ECG " + label_leads[3])
    ax5.set_title("Importance Features "+ label_leads[4] + ", " + diagnosis)
    ax5_.set_title("ECG " + label_leads[4])
    ax6.set_title("Importance Features "+ label_leads[5] + ", " + diagnosis)
    ax6_.set_title("ECG " + label_leads[5])
    ax7.set_title("Importance Features "+ label_leads[6] + ", " + diagnosis)
    ax7_.set_title("ECG " + label_leads[6])
    ax8.set_title("Importance Features "+ label_leads[7] + ", " + diagnosis)
    ax8_.set_title("ECG " + label_leads[7])
    ax9.set_title("Importance Features "+ label_leads[8] + ", " + diagnosis)
    ax9_.set_title("ECG " + label_leads[8])
    ax10.set_title("Importance Features "+ label_leads[9] + ", " + diagnosis)
    ax10_.set_title("ECG " + label_leads[9])
    ax11.set_title("Importance Features "+ label_leads[10] + ", " + diagnosis)
    ax11_.set_title("ECG " + label_leads[10])
    ax12.set_title("Importance Features "+ label_leads[11] + ", " + diagnosis)
    ax12_.set_title("ECG " + label_leads[11])

    plt.show()