import awkward as ak
import hist
import numpy as np
import uproot

# functions creating systematic variations
def jet_pt_resolution(pt):
    # normal distribution with 5% variations, shape matches jets
    counts = ak.num(pt)
    pt_flat = ak.flatten(pt)
    resolution_variation = np.random.normal(np.ones_like(pt_flat), 0.05)
    return ak.unflatten(resolution_variation, counts)

def save_histograms(all_histograms, fileset, filename):
    nominal_samples = [sample for sample in fileset.keys() if "nominal" in sample]

    all_histograms += 1e-6  # add minimal event count to all bins to avoid crashes when processing a small number of samples

    pseudo_data = (all_histograms[:, :, "ttbar", "ME_var"] + all_histograms[:, :, "ttbar", "PS_var"]) / 2  + all_histograms[:, :, "wjets", "nominal"]

    with uproot.recreate(filename) as f:
        for region in ["4j1b", "4j2b"]:
            f[f"{region}_pseudodata"] = pseudo_data[120j::hist.rebin(2), region]
            for sample in nominal_samples:
                sample_name = sample.split("__")[0]
                f[f"{region}_{sample_name}"] = all_histograms[120j::hist.rebin(2), region, sample_name, "nominal"]

                # b-tagging variations
                for i in range(4):
                    for direction in ["up", "down"]:
                        variation_name = f"btag_var_{i}_{direction}"
                        f[f"{region}_{sample_name}_{variation_name}"] = all_histograms[120j::hist.rebin(2), region, sample_name, variation_name]

                # jet energy scale variations
                for variation_name in ["pt_scale_up", "pt_res_up"]:
                    f[f"{region}_{sample_name}_{variation_name}"] = all_histograms[120j::hist.rebin(2), region, sample_name, variation_name]

            # ttbar modeling
            f[f"{region}_ttbar_ME_var"] = all_histograms[120j::hist.rebin(2), region, "ttbar", "ME_var"]
            f[f"{region}_ttbar_PS_var"] = all_histograms[120j::hist.rebin(2), region, "ttbar", "PS_var"]

            f[f"{region}_ttbar_scaledown"] = all_histograms[120j :: hist.rebin(2), region, "ttbar", "scaledown"]
            f[f"{region}_ttbar_scaleup"] = all_histograms[120j :: hist.rebin(2), region, "ttbar", "scaleup"]

            # W+jets scale
            f[f"{region}_wjets_scale_var_down"] = all_histograms[120j :: hist.rebin(2), region, "wjets", "scale_var_down"]
            f[f"{region}_wjets_scale_var_up"] = all_histograms[120j :: hist.rebin(2), region, "wjets", "scale_var_up"]
            
            
def save_ml_histograms(hist_dict, fileset, filename, config):
    nominal_samples = [sample for sample in fileset.keys() if "nominal" in sample]

    
    for feature in config["ml"]["FEATURE_NAMES"]:
        hist_dict[f"hist_{feature}"] += 1e-6  # add minimal event count to all bins to avoid crashes when processing a small number of samples

    with uproot.recreate(filename) as f:
        for feature in config["ml"]["FEATURE_NAMES"]:
            current_hist = hist_dict[f"hist_{feature}"]
            f[f"{feature}_pseudodata"] = (current_hist[:, "ttbar", "ME_var"] + current_hist[:, "ttbar", "PS_var"]) / 2  + current_hist[:, "wjets", "nominal"]
            
            for sample in nominal_samples:
                sample_name = sample.split("__")[0]
                f[f"{feature}_{sample_name}"] = current_hist[:, sample_name, "nominal"]

                # b-tagging variations
                for i in range(4):
                    for direction in ["up", "down"]:
                        variation_name = f"btag_var_{i}_{direction}"
                        f[f"{feature}_{sample_name}_{variation_name}"] = current_hist[:, sample_name, variation_name]

                # jet energy scale variations
                for variation_name in ["pt_scale_up", "pt_res_up"]:
                    f[f"{feature}_{sample_name}_{variation_name}"] = current_hist[:, sample_name, variation_name]

            # ttbar modeling
            f[f"{feature}_ttbar_ME_var"] = current_hist[:, "ttbar", "ME_var"]
            f[f"{feature}_ttbar_PS_var"] = current_hist[:, "ttbar", "PS_var"]

            f[f"{feature}_ttbar_scaledown"] = current_hist[:, "ttbar", "scaledown"]
            f[f"{feature}_ttbar_scaleup"] = current_hist[:, "ttbar", "scaleup"]

            # W+jets scale
            f[f"{feature}_wjets_scale_var_down"] = current_hist[:, "wjets", "scale_var_down"]
            f[f"{feature}_wjets_scale_var_up"] = current_hist[:, "wjets", "scale_var_up"]