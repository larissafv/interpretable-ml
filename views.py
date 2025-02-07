import matplotlib.pyplot as plt
import numpy as np

def plot_counterfactual_ecg(original_ecg, cf_ecg, label_leads = ["DI", "DII", "DIII", "AVL", "AVF", "AVR", "V1", "V2", "V3", "V4", "V5", "V6"]):
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

    ax1.plot(cf_ecg[0], c='r', label='counterfactual')
    ax2.plot(cf_ecg[1], c='r', label='counterfactual')
    ax3.plot(cf_ecg[2], c='r', label='counterfactual')
    ax4.plot(cf_ecg[3], c='r', label='counterfactual')
    ax5.plot(cf_ecg[4], c='r', label='counterfactual')
    ax6.plot(cf_ecg[5], c='r', label='counterfactual')
    ax7.plot(cf_ecg[6], c='r', label='counterfactual')
    ax8.plot(cf_ecg[7], c='r', label='counterfactual')
    ax9.plot(cf_ecg[8], c='r', label='counterfactual')
    ax10.plot(cf_ecg[9], c='r', label='counterfactual')
    ax11.plot(cf_ecg[10], c='r', label='counterfactual')
    ax12.plot(cf_ecg[11], c='r', label='counterfactual')

    ax1.plot(original_ecg[0], c='b', label='original', alpha = 0.65)
    ax2.plot(original_ecg[1], c='b', label='original', alpha = 0.65)
    ax3.plot(original_ecg[2], c='b', label='original', alpha = 0.65)
    ax4.plot(original_ecg[3], c='b', label='original', alpha = 0.65)
    ax5.plot(original_ecg[4], c='b', label='original', alpha = 0.65)
    ax6.plot(original_ecg[5], c='b', label='original', alpha = 0.65)
    ax7.plot(original_ecg[6], c='b', label='original', alpha = 0.65)
    ax8.plot(original_ecg[7], c='b', label='original', alpha = 0.65)
    ax9.plot(original_ecg[8], c='b', label='original', alpha = 0.65)
    ax10.plot(original_ecg[9], c='b', label='original', alpha = 0.65)
    ax11.plot(original_ecg[10], c='b', label='original', alpha = 0.65)
    ax12.plot(original_ecg[11], c='b', label='original', alpha = 0.65)

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

def plot_gs_coefficients(ecg, coeff, diagnosis, label_leads = ["DI", "DII", "DIII", "AVL", "AVF", "AVR", "V1", "V2", "V3", "V4", "V5", "V6"]):
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

def plot_lime_explanation(ecg, top_influential_segments, num_slices, diagnosis, perturb_function=perturb_mean, label_leads = ["DI", "DII", "DIII", "AVL", "AVF", "AVR", "V1", "V2", "V3", "V4", "V5", "V6"]):
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