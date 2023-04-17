import asyncio
import json

import hist
import matplotlib as mpl
import matplotlib.pyplot as plt
import uproot
import awkward as ak
import numpy as np
import os
import mlflow

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def get_client(af="coffea_casa"):
    if af == "coffea_casa":
        from dask.distributed import Client

        client = Client("tls://localhost:8786")

    elif af == "EAF":
        from lpcdaskgateway import LPCGateway

        gateway = LPCGateway()
        cluster = gateway.new_cluster()
        cluster.scale(10)
        print("Please allow up to 60 seconds for HTCondor worker jobs to start")
        print(f"Cluster dashboard: {str(cluster.dashboard_link)}")

        client = cluster.get_client()

    elif af == "local":
        from dask.distributed import Client

        client = Client()

    else:
        raise NotImplementedError(f"unknown analysis facility: {af}")

    return client


def set_style():
    mpl.style.use("ggplot")
    plt.rcParams["axes.facecolor"] = "none"
    plt.rcParams["axes.edgecolor"] = "222222"
    plt.rcParams["axes.labelcolor"] = "222222"
    plt.rcParams["xtick.color"] = "222222"
    plt.rcParams["ytick.color"] = "222222"
    plt.rcParams["font.size"] = 12
    plt.rcParams['text.color'] = "222222"


def construct_fileset(n_files_max_per_sample, use_xcache=False, af_name=""):
    # using https://atlas-groupdata.web.cern.ch/atlas-groupdata/dev/AnalysisTop/TopDataPreparation/XSection-MC15-13TeV.data
    # for reference
    # x-secs are in pb
    xsec_info = {
        "ttbar": 396.87 + 332.97, # nonallhad + allhad, keep same x-sec for all
        "single_top_s_chan": 2.0268 + 1.2676,
        "single_top_t_chan": (36.993 + 22.175)/0.252,  # scale from lepton filter to inclusive
        "single_top_tW": 37.936 + 37.906,
        "wjets": 61457 * 0.252,  # e/mu+nu final states
        "data": None
    }

    # list of files
    with open("nanoaod_inputs.json") as f:
        file_info = json.load(f)

    # process into "fileset" summarizing all info
    fileset = {}
    for process in file_info.keys():
        if process == "data":
            continue  # skip data

        for variation in file_info[process].keys():
            file_list = file_info[process][variation]["files"]
            if n_files_max_per_sample != -1:
                file_list = file_list[:n_files_max_per_sample]  # use partial set of samples

            file_paths = [f["path"] for f in file_list]
            if use_xcache:
                file_paths = [f.replace("https://xrootd-local.unl.edu:1094", "root://red-xcache1.unl.edu") for f in file_paths]
            if af_name == "ssl-dev":
                # point to local files on /data
                file_paths = [f.replace("https://xrootd-local.unl.edu:1094//store/user/", "/data/alheld/") for f in file_paths]
            nevts_total = sum([f["nevts"] for f in file_list])
            metadata = {"process": process, "variation": variation, "nevts": nevts_total, "xsec": xsec_info[process]}
            fileset.update({f"{process}__{variation}": {"files": file_paths, "metadata": metadata}})

    return fileset

def save_ml_histograms(all_histograms, fileset, filename):
    
    features = ["deltar_leptontoplep","deltar_w1w2","deltar_w1tophad","deltar_w2tophad","mass_leptontoplep","mass_w1w2",
                "mass_w1w2tophad","pt_w1w2tophad","pt_w1","pt_w2","pt_tophad","pt_toplep",
                "btag_w1","btag_w2","btag_tophad","btag_toplep","qgl_w1","qgl_w2","qgl_tophad","qgl_toplep"]
    
    nominal_samples = [sample for sample in fileset.keys() if "nominal" in sample]
    
    with uproot.recreate(filename) as f:
        
        for feature in features:
        
            # add minimal event count to all bins to avoid crashes when processing a small number of samples
            all_histograms[f"hist_{feature}"] += 1e-6 
            
            f[f"{feature}_pseudodata"] = ((all_histograms[f"hist_{feature}"][:, "ttbar", "ME_var"] 
                                           + all_histograms[f"hist_{feature}"][:, "ttbar", "PS_var"]) / 2  
                                          + all_histograms[f"hist_{feature}"][:, "wjets", "nominal"]).project("observable")
        
            for sample in nominal_samples:
                sample_name = sample.split("__")[0]
                f[f"{feature}_{sample_name}"] = all_histograms[f"hist_{feature}"][:, sample_name, "nominal"].project("observable")
                # b-tagging variations
                for i in range(4):
                    for direction in ["up", "down"]:
                        variation_name = f"btag_var_{i}_{direction}"
                        f[f"{feature}_{sample_name}_{variation_name}"] = all_histograms[f"hist_{feature}"][
                            :, 
                            sample_name, 
                            variation_name
                        ].project("observable")
                        
                # jet energy scale variations
                for variation_name in ["pt_scale_up", "pt_res_up"]:
                    f[f"{feature}_{sample_name}_{variation_name}"] = all_histograms[f"hist_{feature}"][
                        :, 
                        sample_name,                                                           
                        variation_name
                    ].project("observable")


                f[f"{feature}_ttbar_ME_var"] = all_histograms[f"hist_{feature}"][:, "ttbar", "ME_var"].project("observable")
                f[f"{feature}_ttbar_PS_var"] = all_histograms[f"hist_{feature}"][:, "ttbar", "PS_var"].project("observable")
                f[f"{feature}_ttbar_scaledown"] = all_histograms[f"hist_{feature}"][:, "ttbar", "scaledown"].project("observable")
                f[f"{feature}_ttbar_scaleup"] = all_histograms[f"hist_{feature}"][:, "ttbar", "scaleup"].project("observable")

                # W+jets scale
                f[f"{feature}_wjets_scale_var_down"] = all_histograms[f"hist_{feature}"][:, "wjets", 
                                                                                         "scale_var_down"].project("observable")
                f[f"{feature}_wjets_scale_var_up"] = all_histograms[f"hist_{feature}"][:, "wjets", 
                                                                                       "scale_var_up"].project("observable")


def save_histograms(all_histograms, fileset, filename):
    nominal_samples = [sample for sample in fileset.keys() if "nominal" in sample]

    all_histograms += 1e-6  # add minimal event count to all bins to avoid crashes when processing a small number of samples

    pseudo_data = (all_histograms[:, :, "ttbar", "ME_var"] + all_histograms[:, :, "ttbar", "PS_var"]) / 2  + all_histograms[:, :, "wjets", "nominal"]

    with uproot.recreate(filename) as f:
                
        for region in ["4j1b", "4j2b"]:
            f[f"{region}_pseudodata"] = pseudo_data[120j::hist.rebin(2), 
                                                    region].project("observable")
            for sample in nominal_samples:
                sample_name = sample.split("__")[0]
                f[f"{region}_{sample_name}"] = all_histograms[120j::hist.rebin(2), 
                                                              region, 
                                                              sample_name, 
                                                              "nominal"].project("observable")
                
                # b-tagging variations
                for i in range(4):
                    for direction in ["up", "down"]:
                        variation_name = f"btag_var_{i}_{direction}"
                        f[f"{region}_{sample_name}_{variation_name}"] = all_histograms[120j::hist.rebin(2), 
                                                                                       region, 
                                                                                       sample_name, 
                                                                                       variation_name].project("observable")

                # jet energy scale variations
                for variation_name in ["pt_scale_up", "pt_res_up"]:
                    f[f"{region}_{sample_name}_{variation_name}"] = all_histograms[120j::hist.rebin(2), 
                                                                                   region, 
                                                                                   sample_name, 
                                                                                   variation_name].project("observable")
                    

            f[f"{region}_ttbar_ME_var"] = all_histograms[120j::hist.rebin(2), 
                                                         region, 
                                                         "ttbar", 
                                                         "ME_var"].project("observable")
            f[f"{region}_ttbar_PS_var"] = all_histograms[120j::hist.rebin(2), 
                                                         region, 
                                                         "ttbar", 
                                                         "PS_var"].project("observable")

            f[f"{region}_ttbar_scaledown"] = all_histograms[120j::hist.rebin(2), 
                                                            region, 
                                                            "ttbar", 
                                                            "scaledown"].project("observable")
            f[f"{region}_ttbar_scaleup"] = all_histograms[120j::hist.rebin(2), 
                                                          region, 
                                                          "ttbar", 
                                                          "scaleup"].project("observable")
            

            # W+jets scale
            f[f"{region}_wjets_scale_var_down"] = all_histograms[120j::hist.rebin(2), 
                                                                 region, 
                                                                 "wjets", 
                                                                 "scale_var_down"].project("observable")
            f[f"{region}_wjets_scale_var_up"] = all_histograms[120j::hist.rebin(2), 
                                                               region, 
                                                               "wjets", 
                                                               "scale_var_up"].project("observable")
                
            
            
def get_permutations_dict(MAX_N_JETS, include_labels=False, include_eval_mat=False):
    
    # calculate the dictionary of permutations for each number of jets
    permutations_dict = {}
    for n in range(4,MAX_N_JETS+1):
        test = ak.Array(range(n))
        unzipped = ak.unzip(ak.argcartesian([test]*4,axis=0))

        combos = ak.combinations(ak.Array(range(4)), 2, axis=0)
        different = unzipped[combos[0]["0"]]!=unzipped[combos[0]["1"]]
        for i in range(1,len(combos)):
            different = different & (unzipped[combos[i]["0"]]!=unzipped[combos[i]["1"]])

        permutations = ak.zip([test[unzipped[i][different]] for i in range(len(unzipped))],
                              depth_limit=1).tolist()


        permutations = ak.concatenate([test[unzipped[i][different]][..., np.newaxis] 
                                       for i in range(len(unzipped))], 
                                      axis=1).to_list()

        permutations_dict[n] = permutations

    # for each permutation, calculate the corresponding label
    labels_dict = {}
    for n in range(4,MAX_N_JETS+1):

        current_labels = []
        for inds in permutations_dict[n]:

            inds = np.array(inds)
            current_label = 100*np.ones(n)
            current_label[inds[:2]] = 24
            current_label[inds[2]] = 6
            current_label[inds[3]] = -6
            current_labels.append(current_label.tolist())

        labels_dict[n] = current_labels

    # get rid of duplicates since we consider W jets to be exchangeable
    # (halves the number of permutations we consider)
    for n in range(4,MAX_N_JETS+1):
        res = []
        for idx, val in enumerate(labels_dict[n]):
            if val in labels_dict[n][:idx]:
                res.append(idx)
        labels_dict[n] = np.array(labels_dict[n])[res].tolist()
        permutations_dict[n] = np.array(permutations_dict[n])[res].tolist()
        print("number of permutations for n=",n,": ", len(permutations_dict[n]))
        
    if include_labels and not include_eval_mat:
        return permutations_dict, labels_dict
    
    elif include_labels and include_eval_mat:
        # these matrices tell you the overlap between the predicted label (rows) and truth label (columns)
        # the "score" in each matrix entry is the number of jets which are assigned correctly        
        evaluation_matrices = {} # overall event score

        for n in range(4, MAX_N_JETS+1):
            evaluation_matrix = np.zeros((len(permutations_dict[n]),len(permutations_dict[n])))

            for i in range(len(permutations_dict[n])):
                for j in range(len(permutations_dict[n])):
                    evaluation_matrix[i,j]=sum(np.equal(labels_dict[n][i], labels_dict[n][j]))

            evaluation_matrices[n] = evaluation_matrix/4
            print("calculated evaluation matrix for n=",n)
            
        return permutations_dict, labels_dict, evaluation_matrices
    
    else:
        return permutations_dict

    
# function to provide necessary environment variables to workers
def initialize_mlflow(): 
    
    os.system("pip install boto3")

    os.environ['MLFLOW_TRACKING_URI'] = "https://mlflow.software-dev.ncsa.cloud"
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = "https://mlflow-minio-api.software-dev.ncsa.cloud"
    os.environ['AWS_ACCESS_KEY_ID'] = "bengal1"
    os.environ['AWS_SECRET_ACCESS_KEY'] = "leftfoot1"
    
    mlflow.set_tracking_uri('https://mlflow.software-dev.ncsa.cloud') 
    mlflow.set_experiment("agc-demo")
    
    
import particle
from anytree import NodeMixin, RenderTree, Node

class GenPartNode(NodeMixin):
    def __init__(self, name, genPart, parent=None, children=None):
        super(GenPartNode, self).__init__()
        self.name = name
        self.genPart = genPart
        self.parent = parent
        if children:
            self.children = children

def printTrees(particles):
    
    origins = []
    for genpart in particles:
        if genpart.genPartIdxMother==-1:
            
            # create origin node
            origin = GenPartNode(particle.Particle.from_pdgid(genpart.pdgId).name, genpart)
            origins.append(origin)

            # initialize lists/queues to keep track
            queue_node = []
            visited_genpart = []
            queue_genpart = []

            # add origin particle/node to queue/visited
            queue_node.append(origin)
            visited_genpart.append((genpart.pdgId,genpart.pt,genpart.eta,genpart.phi))
            queue_genpart.append(genpart)
            
            # loop through queue
            while queue_genpart:
            
                # grab top elements from queue
                g = queue_genpart.pop(0)
                n = queue_node.pop(0)

                # iterate through daughters
                for daughter in g.children:

                    # (should be) unique id for particle
                    daughter_tuple = (daughter.pdgId,daughter.pt,daughter.eta,daughter.phi)

                    # if we have not visited particle yet
                    if daughter_tuple not in visited_genpart:
                        
                        # add to queue
                        visited_genpart.append(daughter_tuple)
                        queue_genpart.append(daughter)

                        # create new node
                        node =  GenPartNode(particle.Particle.from_pdgid(daughter.pdgId).name, 
                                            daughter,
                                            parent = n)
                        
                        queue_node.append(node)
                                
        
    # printing trees
    for origin in origins:
        for pre, fill, node in RenderTree(origin):
            print("%s%s" % (pre, node.name))
            
    return



## get inputs to BDT
def get_features(jets, electrons, muons, permutations_dict):
    '''
    Calculate features for each of the 12 combinations per event
    
    Args:
        jets: selected jets
        electrons: selected electrons
        muons: selected muons
        permutations_dict: which permutations to consider for each number of jets in an event
    
    Returns:
        features (flattened to remove event level)
        perm_counts: how many permutations in each event. use to unflatten features
    '''
    
    # calculate number of jets in each event
    njet = ak.num(jets).to_numpy()
    # don't consider every jet for events with high jet multiplicity
    njet[njet>max(permutations_dict.keys())] = max(permutations_dict.keys())
    # create awkward array of permutation indices
    perms = ak.Array([permutations_dict[n] for n in njet])
    perm_counts = ak.num(perms)
    
    
    #### calculate features ####
    #### calculate features ####
    features = np.zeros((sum(perm_counts),20))
    
    # grab lepton info
    leptons = ak.flatten(ak.concatenate((electrons, muons),axis=1),axis=-1)

    feature_count = 0
    
    # delta R between top1 and lepton
    features[:,0] = ak.flatten(np.sqrt((leptons.eta - jets[perms[...,3]].eta)**2 + 
                                       (leptons.phi - jets[perms[...,3]].phi)**2)).to_numpy()

    
    #delta R between the two W
    features[:,1] = ak.flatten(np.sqrt((jets[perms[...,0]].eta - jets[perms[...,1]].eta)**2 + 
                                       (jets[perms[...,0]].phi - jets[perms[...,1]].phi)**2)).to_numpy()

    #delta R between W and top2
    features[:,2] = ak.flatten(np.sqrt((jets[perms[...,0]].eta - jets[perms[...,2]].eta)**2 + 
                                       (jets[perms[...,0]].phi - jets[perms[...,2]].phi)**2)).to_numpy()
    features[:,3] = ak.flatten(np.sqrt((jets[perms[...,1]].eta - jets[perms[...,2]].eta)**2 + 
                                       (jets[perms[...,1]].phi - jets[perms[...,2]].phi)**2)).to_numpy()

    # combined mass of top1 and lepton
    features[:,4] = ak.flatten((leptons + jets[perms[...,3]]).mass).to_numpy()

    # combined mass of W
    features[:,5] = ak.flatten((jets[perms[...,0]] + jets[perms[...,1]]).mass).to_numpy()

    # combined mass of W and top2
    features[:,6] = ak.flatten((jets[perms[...,0]] + jets[perms[...,1]] + 
                                 jets[perms[...,2]]).mass).to_numpy()
    
    feature_count+=1
    # combined pT of W and top2
    features[:,7] = ak.flatten((jets[perms[...,0]] + jets[perms[...,1]] + 
                                 jets[perms[...,2]]).pt).to_numpy()


    # pt of every jet
    features[:,8] = ak.flatten(jets[perms[...,0]].pt).to_numpy()
    features[:,9] = ak.flatten(jets[perms[...,1]].pt).to_numpy()
    features[:,10] = ak.flatten(jets[perms[...,2]].pt).to_numpy()
    features[:,11] = ak.flatten(jets[perms[...,3]].pt).to_numpy()

    # btagCSVV2 of every jet
    features[:,12] = ak.flatten(jets[perms[...,0]].btagCSVV2).to_numpy()
    features[:,13] = ak.flatten(jets[perms[...,1]].btagCSVV2).to_numpy()
    features[:,14] = ak.flatten(jets[perms[...,2]].btagCSVV2).to_numpy()
    features[:,15] = ak.flatten(jets[perms[...,3]].btagCSVV2).to_numpy()
    
    # qgl of every jet
    features[:,16] = ak.flatten(jets[perms[...,0]].qgl).to_numpy()
    features[:,17] = ak.flatten(jets[perms[...,1]].qgl).to_numpy()
    features[:,18] = ak.flatten(jets[perms[...,2]].qgl).to_numpy()
    features[:,19] = ak.flatten(jets[perms[...,3]].qgl).to_numpy()

    return features, perm_counts

def plot_data_mc(model_prediction_prefit, model_prediction_postfit, data, config):
    '''
    Display data-MC comparison plots. Modelled using cabinetry's visualize.data_mc but displays in a less cumbersome way (and shows prefit and postfit plots alongside one another.
    
    Args:
        model_prediction_prefit (model_utils.ModelPrediction): prefit model prediction to show
        model_prediction_postfit (model_utils.ModelPrediction): postfit model prediction to show
        data (List[float]): data to include in visualization
        config (Optional[Dict[str, Any]], optional): cabinetry configuration needed for binning and axis labels
    
    Returns:
        figs: list of matplotlib Figures containing prefit and postfit plots 
    '''
    
    n_bins_total = sum(model_prediction_prefit.model.config.channel_nbins.values())
    if len(data) != n_bins_total:
        data = data[:n_bins_total]

    data_yields = [data[model_prediction_prefit.model.config.channel_slices[ch]] 
                   for ch in model_prediction_prefit.model.config.channels]
    
    channels = model_prediction_prefit.model.config.channels
    channel_indices = [model_prediction_prefit.model.config.channels.index(ch) for ch in channels]
    
    figs = []
    for i_chan, channel_name in zip(channel_indices, channels):
        histogram_dict_list_prefit = []  # one dict per region/channel
        histogram_dict_list_postfit = []  # one dict per region/channel

        regions = [reg for reg in config["Regions"] if reg["Name"] == channel_name]
        region_dict = regions[0]
        bin_edges = np.asarray(region_dict["Binning"])
        variable = region_dict.get("Variable", "bin")
        
        for i_sam, sample_name in enumerate(model_prediction_prefit.model.config.samples):
            histogram_dict_list_prefit.append(
                {"label": sample_name, "isData": False, 
                 "yields": model_prediction_prefit.model_yields[i_chan][i_sam], "variable": variable}
            )
            histogram_dict_list_postfit.append(
                {"label": sample_name, "isData": False, 
                 "yields": model_prediction_postfit.model_yields[i_chan][i_sam], "variable": variable}
            )
            
        # add data sample
        histogram_dict_list_prefit.append(
            {"label": "Data", "isData": True, "yields": data_yields[i_chan], "variable": variable}
        )
        histogram_dict_list_postfit.append(
            {"label": "Data", "isData": True, "yields": data_yields[i_chan], "variable": variable}
        )
        
        label_prefit = f"{channel_name}\n{model_prediction_prefit.label}"
        label_postfit = f"{channel_name}\n{model_prediction_postfit.label}"

        mc_histograms_yields_prefit = []
        mc_histograms_yields_postfit = []
        mc_labels = []
        for i in range(len(histogram_dict_list_prefit)):
            if histogram_dict_list_prefit[i]["isData"]:
                data_histogram_yields = histogram_dict_list_prefit[i]["yields"]
                data_histogram_stdev = np.sqrt(data_histogram_yields)
                data_label = histogram_dict_list_prefit[i]["label"]
            else:
                mc_histograms_yields_prefit.append(histogram_dict_list_prefit[i]["yields"])
                mc_histograms_yields_postfit.append(histogram_dict_list_postfit[i]["yields"])
                mc_labels.append(histogram_dict_list_prefit[i]["label"])
                
        mpl.style.use("seaborn-v0_8-colorblind")
    
        fig = plt.figure(figsize=(12, 6), layout="constrained")
        gs = fig.add_gridspec(nrows=2, ncols=2, hspace=0, height_ratios=[3, 1])
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[1,0])
        ax3 = fig.add_subplot(gs[0,1])
        ax4 = fig.add_subplot(gs[1,1])

        # increase font sizes
        for item in (
            [ax1.yaxis.label, ax2.xaxis.label, ax2.yaxis.label, ax3.yaxis.label, ax4.xaxis.label, ax4.yaxis.label]
            + ax1.get_yticklabels() + ax2.get_xticklabels() + ax2.get_yticklabels()
            + ax3.get_yticklabels() + ax4.get_xticklabels() + ax4.get_yticklabels()
        ):
            item.set_fontsize("large")
        
        # plot MC stacked together
        total_yield_prefit = np.zeros_like(mc_histograms_yields_prefit[0])
        total_yield_postfit = np.zeros_like(mc_histograms_yields_postfit[0])

        bin_right_edges = bin_edges[1:]
        bin_left_edges = bin_edges[:-1]
        bin_width = bin_right_edges - bin_left_edges
        bin_centers = 0.5 * (bin_left_edges + bin_right_edges)

        mc_containers_prefit = []
        mc_containers_postfit = []
        
        for i in range(len(mc_histograms_yields_prefit)):
            mc_container = ax1.bar(bin_centers, mc_histograms_yields_prefit[i], width=bin_width, bottom=total_yield_prefit)
            mc_containers_prefit.append(mc_container)
            mc_container = ax3.bar(bin_centers, mc_histograms_yields_postfit[i], width=bin_width, bottom=total_yield_postfit)
            mc_containers_postfit.append(mc_container)

            # add a black line on top of each sample
            line_x = [y for y in bin_edges for _ in range(2)][1:-1]
            line_y = [y for y in (mc_histograms_yields_prefit[i] + total_yield_prefit) for _ in range(2)]
            ax1.plot(line_x, line_y, "-", color="black", linewidth=0.5)
            line_y = [y for y in (mc_histograms_yields_postfit[i] + total_yield_postfit) for _ in range(2)]
            ax3.plot(line_x, line_y, "-", color="black", linewidth=0.5)

            total_yield_prefit += mc_histograms_yields_prefit[i]
            total_yield_postfit += mc_histograms_yields_postfit[i]
            
        # add total MC uncertainty
        total_model_unc_prefit = np.asarray(model_prediction_prefit.total_stdev_model_bins[i_chan][-1])
        mc_unc_container_prefit = ax1.bar(bin_centers, 2*total_model_unc_prefit, width=bin_width, 
                                          bottom=total_yield_prefit-total_model_unc_prefit, fill=False, linewidth=0, 
                                          edgecolor="gray", hatch=3 * "/",)
        total_model_unc_postfit = np.asarray(model_prediction_postfit.total_stdev_model_bins[i_chan][-1])
        mc_unc_container_postfit = ax3.bar(bin_centers, 2*total_model_unc_postfit, width=bin_width, 
                                           bottom=total_yield_postfit-total_model_unc_postfit, fill=False, linewidth=0, 
                                           edgecolor="gray", hatch=3 * "/",)
    
        # plot data
        data_container_prefit = ax1.errorbar(bin_centers, data_histogram_yields, yerr=data_histogram_stdev, fmt="o", color="k")
        data_container_postfit = ax3.errorbar(bin_centers, data_histogram_yields, yerr=data_histogram_stdev, fmt="o", color="k")
        
        # ratio plots
        ax2.plot([bin_left_edges[0], bin_right_edges[-1]], [1, 1], "--", color="black", linewidth=1)  # reference line along y=1
        ax4.plot([bin_left_edges[0], bin_right_edges[-1]], [1, 1], "--", color="black", linewidth=1)  # reference line along y=1

        n_zero_pred_prefit = sum(total_yield_prefit == 0.0)  # number of bins with zero predicted yields
        if n_zero_pred_prefit > 0:
            log.warning(f"(PREFIT) predicted yield is zero in {n_zero_pred_prefit} bin(s), excluded from ratio plot")
        nonzero_model_yield_prefit = total_yield_prefit != 0.0
        if np.any(total_yield_prefit < 0.0):
            raise ValueError(f"(PREFIT) {label_prefit} total model yield has negative bin(s): {total_yield_prefit.tolist()}")
        
        n_zero_pred_postfit = sum(total_yield_postfit == 0.0)  # number of bins with zero predicted yields
        if n_zero_pred_postfit > 0:
            log.warning(f"(POSTFIT) predicted yield is zero in {n_zero_pred_postfit} bin(s), excluded from ratio plot")
        nonzero_model_yield_postfit = total_yield_postfit != 0.0
        if np.any(total_yield_postfit < 0.0):
            raise ValueError(f"(POSTFIT) {label_postfit} total model yield has negative bin(s): {total_yield_postfit.tolist()}")
        
        # add uncertainty band around y=1
        rel_mc_unc_prefit = total_model_unc_prefit / total_yield_prefit
        rel_mc_unc_postfit = total_model_unc_postfit / total_yield_postfit
        # do not show band in bins where total model yield is 0
        ax2.bar(bin_centers[nonzero_model_yield_prefit], 2*rel_mc_unc_prefit[nonzero_model_yield_prefit], 
                width=bin_width[nonzero_model_yield_prefit], bottom=1.0-rel_mc_unc_prefit[nonzero_model_yield_prefit], 
                fill=False, linewidth=0, edgecolor="gray", hatch=3 * "/")
        ax4.bar(bin_centers[nonzero_model_yield_postfit], 2*rel_mc_unc_postfit[nonzero_model_yield_postfit], 
                width=bin_width[nonzero_model_yield_postfit], bottom=1.0-rel_mc_unc_postfit[nonzero_model_yield_postfit], 
                fill=False, linewidth=0, edgecolor="gray", hatch=3 * "/")
        
        # data in ratio plot
        data_model_ratio_prefit = data_histogram_yields / total_yield_prefit
        data_model_ratio_unc_prefit = data_histogram_stdev / total_yield_prefit
        data_model_ratio_postfit = data_histogram_yields / total_yield_postfit
        data_model_ratio_unc_postfit = data_histogram_stdev / total_yield_postfit
        # mask data in bins where total model yield is 0
        ax2.errorbar(bin_centers[nonzero_model_yield_prefit], data_model_ratio_prefit[nonzero_model_yield_prefit], 
                     yerr=data_model_ratio_unc_prefit[nonzero_model_yield_prefit], fmt="o", color="k")
        ax4.errorbar(bin_centers[nonzero_model_yield_postfit], data_model_ratio_postfit[nonzero_model_yield_postfit], 
                     yerr=data_model_ratio_unc_postfit[nonzero_model_yield_postfit], fmt="o", color="k")
        
        # get the highest single bin yield, from the sum of MC or data
        y_max = max(max(np.amax(total_yield_prefit), np.amax(data_histogram_yields)),
                    max(np.amax(total_yield_postfit), np.amax(data_histogram_yields)))

        ax1.set_ylim([0, y_max * 1.5])  # 50% headroom
        ax3.set_ylim([0, y_max * 1.5])  # 50% headroom
    
        # figure label (region name)
        at = mpl.offsetbox.AnchoredText(label_prefit, loc="upper left", frameon=False, prop={"fontsize": "large", "linespacing": 1.5})
        ax1.add_artist(at)
        at = mpl.offsetbox.AnchoredText(label_postfit, loc="upper left", frameon=False, prop={"fontsize": "large", "linespacing": 1.5})
        ax3.add_artist(at)
        
        # MC contributions in inverse order, such that first legend entry corresponds to
        # the last (highest) contribution to the stack
        all_containers = mc_containers_prefit[::-1] + [mc_unc_container_prefit, data_container_prefit]
        all_labels = mc_labels[::-1] + ["Uncertainty", data_label]
        ax1.legend(all_containers, all_labels, frameon=False, fontsize="large", loc="upper right")
        all_containers = mc_containers_postfit[::-1] + [mc_unc_container_postfit, data_container_postfit]
        ax3.legend(all_containers, all_labels, frameon=False, fontsize="large", loc="upper right")
        
        ax1.set_xlim(bin_edges[0], bin_edges[-1])
        ax1.set_ylabel("events")
        ax1.set_xticklabels([])
        ax1.set_xticklabels([], minor=True)
        ax1.tick_params(axis="both", which="major", pad=8)  # tick label - axis padding
        ax1.tick_params(direction="in", top=True, right=True, which="both")

        ax3.set_xlim(bin_edges[0], bin_edges[-1])
        ax3.set_xticklabels([])
        ax3.set_xticklabels([], minor=True)
        ax3.set_yticklabels([])
        ax3.set_yticklabels([], minor=True)
        ax3.tick_params(axis="both", which="major", pad=8)  # tick label - axis padding
        ax3.tick_params(direction="in", top=True, right=True, which="both")

        ax2.set_xlim(bin_edges[0], bin_edges[-1])
        ax2.set_ylim([0.5, 1.5])
        ax2.set_xlabel(histogram_dict_list_prefit[0]["variable"])
        ax2.set_ylabel("data / model")
        ax2.set_yticks([0.5, 0.75, 1.0, 1.25, 1.5])
        ax2.set_yticklabels([0.5, 0.75, 1.0, 1.25, ""])
        ax2.tick_params(axis="both", which="major", pad=8)
        ax2.tick_params(direction="in", top=True, right=True, which="both")

        ax4.set_xlim(bin_edges[0], bin_edges[-1])
        ax4.set_ylim([0.5, 1.5])
        ax4.set_xlabel(histogram_dict_list_postfit[0]["variable"])
        ax4.set_yticklabels([])
        ax4.set_yticklabels([], minor=True)
        ax4.tick_params(axis="both", which="major", pad=8)
        ax4.tick_params(direction="in", top=True, right=True, which="both")
        
        fig.show()

        figs.append({"figure": fig, "region": channel_name})
    
    
    return figs
    