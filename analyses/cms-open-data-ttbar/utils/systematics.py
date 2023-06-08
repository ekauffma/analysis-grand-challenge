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


ttbar_variations = ["ME_var", "PS_var", "scaleup", "scaledown"]
wjets_variations = ["scale_var_down", "scale_var_up"]


def save_histograms(hist_dict, fileset, filename, channel_names, rebin=True):
    nominal_samples = [sample for sample in fileset.keys() if "nominal" in sample]

    for channel in channel_names:
        hist_dict[
            channel
        ] += 1e-6  # add minimal event count to all bins to avoid crashes when processing a small number of samples

    with uproot.recreate(filename) as f:
        out_dict = {}
        for channel in channel_names:
            current_hist = hist_dict[channel]
            out_dict[f"{channel}_pseudodata"] = (
                current_hist[:, "ttbar", "ME_var"] + current_hist[:, "ttbar", "PS_var"]
            ) / 2 + current_hist[:, "wjets", "nominal"]

            for sample in nominal_samples:
                sample_name = sample.split("__")[0]
                out_dict[f"{channel}_{sample_name}"] = current_hist[
                    :, sample_name, "nominal"
                ]

                # b-tagging variations
                for i in range(4):
                    for direction in ["up", "down"]:
                        variation_name = f"btag_var_{i}_{direction}"
                        out_dict[
                            f"{channel}_{sample_name}_{variation_name}"
                        ] = current_hist[:, sample_name, variation_name]

                # jet energy scale variations
                for variation_name in ["pt_scale_up", "pt_res_up"]:
                    out_dict[
                        f"{channel}_{sample_name}_{variation_name}"
                    ] = current_hist[:, sample_name, variation_name]

            # ttbar modeling
            for var in ttbar_variations:
                out_dict[f"{channel}_ttbar_{var}"] = current_hist[:, "ttbar", var]

            # W+jets scale
            for var in wjets_variations:
                out_dict[f"{channel}_wjets_{var}"] = current_hist[:, "wjets", var]

        # write to file and rebin if necessary
        for key in out_dict.keys():
            if rebin:
                f[key] = out_dict[key][120j :: hist.rebin(2)]
            else:
                f[key] = out_dict[key]
