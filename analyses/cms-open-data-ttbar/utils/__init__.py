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


def save_histograms(all_histograms, fileset, filename, ml=False):
    nominal_samples = [sample for sample in fileset.keys() if "nominal" in sample]

    all_histograms += 1e-6  # add minimal event count to all bins to avoid crashes when processing a small number of samples

    pseudo_data = (all_histograms[:, :, :, "ttbar", "ME_var"] + all_histograms[:, :, :, "ttbar", "PS_var"]) / 2  + all_histograms[:, :, :, "wjets", "nominal"]

    with uproot.recreate(filename) as f:
        
        region_list = ["4j1b", "4j2b"]
        if ml:
            region_list.append("4j2bML")
        
        for region in region_list:
            if not region=="4j2bML":
                f[f"{region}_pseudodata"] = pseudo_data[120j::hist.rebin(2), 
                                                        120j::hist.rebin(2), 
                                                        region].project("observable")
                for sample in nominal_samples:
                    sample_name = sample.split("__")[0]
                    f[f"{region}_{sample_name}"] = all_histograms[120j::hist.rebin(2), 
                                                                  120j::hist.rebin(2), 
                                                                  region, 
                                                                  sample_name, 
                                                                  "nominal"].project("observable")
                
                    # b-tagging variations
                    for i in range(4):
                        for direction in ["up", "down"]:
                            variation_name = f"btag_var_{i}_{direction}"
                            f[f"{region}_{sample_name}_{variation_name}"] = all_histograms[120j::hist.rebin(2), 
                                                                                           120j::hist.rebin(2), 
                                                                                           region, 
                                                                                           sample_name, 
                                                                                           variation_name].project("observable")

                    # jet energy scale variations
                    for variation_name in ["pt_scale_up", "pt_res_up"]:
                        f[f"{region}_{sample_name}_{variation_name}"] = all_histograms[120j::hist.rebin(2), 
                                                                                       120j::hist.rebin(2), 
                                                                                       region, 
                                                                                       sample_name, 
                                                                                       variation_name].project("observable")
                    

                f[f"{region}_ttbar_ME_var"] = all_histograms[120j::hist.rebin(2), 
                                                             120j::hist.rebin(2), 
                                                             region, 
                                                             "ttbar", 
                                                             "ME_var"].project("observable")
                f[f"{region}_ttbar_PS_var"] = all_histograms[120j::hist.rebin(2), 
                                                             120j::hist.rebin(2), 
                                                             region, 
                                                             "ttbar", 
                                                             "PS_var"].project("observable")

                f[f"{region}_ttbar_scaledown"] = all_histograms[120j::hist.rebin(2), 
                                                                120j::hist.rebin(2), 
                                                                region, 
                                                                "ttbar", 
                                                                "scaledown"].project("observable")
                f[f"{region}_ttbar_scaleup"] = all_histograms[120j::hist.rebin(2), 
                                                              120j::hist.rebin(2), 
                                                              region, 
                                                              "ttbar", 
                                                              "scaleup"].project("observable")
            

                # W+jets scale
                f[f"{region}_wjets_scale_var_down"] = all_histograms[120j::hist.rebin(2), 
                                                                     120j::hist.rebin(2), 
                                                                     region, 
                                                                     "wjets", 
                                                                     "scale_var_down"].project("observable")
                f[f"{region}_wjets_scale_var_up"] = all_histograms[120j::hist.rebin(2), 
                                                                   120j::hist.rebin(2), 
                                                                   region, 
                                                                   "wjets", 
                                                                   "scale_var_up"].project("observable")
                
            else:
                f[f"{region}_pseudodata"] = pseudo_data[120j::hist.rebin(2), 
                                                        120j::hist.rebin(2), 
                                                        "4j2b"].project("ml_observable")
                for sample in nominal_samples:
                    sample_name = sample.split("__")[0]
                    f[f"{region}_{sample_name}"] = all_histograms[120j::hist.rebin(2), 
                                                                  120j::hist.rebin(2), 
                                                                  "4j2b", 
                                                                  sample_name, 
                                                                  "nominal"].project("ml_observable")
                
                    # b-tagging variations
                    for i in range(4):
                        for direction in ["up", "down"]:
                            variation_name = f"btag_var_{i}_{direction}"
                            f[f"{region}_{sample_name}_{variation_name}"] = all_histograms[120j::hist.rebin(2), 
                                                                                           120j::hist.rebin(2), 
                                                                                           "4j2b", 
                                                                                           sample_name, 
                                                                                           variation_name].project("ml_observable")

                    # jet energy scale variations
                    for variation_name in ["pt_scale_up", "pt_res_up"]:
                        f[f"{region}_{sample_name}_{variation_name}"] = all_histograms[120j::hist.rebin(2), 
                                                                                       120j::hist.rebin(2), 
                                                                                       "4j2b", 
                                                                                       sample_name, 
                                                                                       variation_name].project("ml_observable")
                    

                f[f"{region}_ttbar_ME_var"] = all_histograms[120j::hist.rebin(2), 
                                                             120j::hist.rebin(2), 
                                                             "4j2b", 
                                                             "ttbar", 
                                                             "ME_var"].project("ml_observable")
                f[f"{region}_ttbar_PS_var"] = all_histograms[120j::hist.rebin(2), 
                                                             120j::hist.rebin(2), 
                                                             "4j2b", 
                                                             "ttbar", 
                                                             "PS_var"].project("ml_observable")

                f[f"{region}_ttbar_scaledown"] = all_histograms[120j::hist.rebin(2), 
                                                                120j::hist.rebin(2), 
                                                                "4j2b", 
                                                                "ttbar", 
                                                                "scaledown"].project("ml_observable")
                f[f"{region}_ttbar_scaleup"] = all_histograms[120j::hist.rebin(2), 
                                                              120j::hist.rebin(2), 
                                                              "4j2b", 
                                                              "ttbar", 
                                                              "scaleup"].project("ml_observable")
            

                # W+jets scale
                f[f"{region}_wjets_scale_var_down"] = all_histograms[120j::hist.rebin(2), 
                                                                     120j::hist.rebin(2), 
                                                                     "4j2b", 
                                                                     "wjets", 
                                                                     "scale_var_down"].project("ml_observable")
                f[f"{region}_wjets_scale_var_up"] = all_histograms[120j::hist.rebin(2), 
                                                                   120j::hist.rebin(2), 
                                                                   "4j2b", 
                                                                   "wjets", 
                                                                   "scale_var_up"].project("ml_observable")
            
            
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