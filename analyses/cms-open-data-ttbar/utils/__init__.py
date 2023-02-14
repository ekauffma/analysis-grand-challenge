import asyncio
import json

import hist
import matplotlib as mpl
import matplotlib.pyplot as plt
import uproot

from func_adl_servicex import ServiceXSourceUpROOT
from func_adl import ObjectStream
from coffea.processor import servicex
from servicex import ServiceXDataset

import particle
from anytree import NodeMixin, RenderTree, Node

import awkward as ak
import numpy as np


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


def construct_fileset(n_files_max_per_sample, use_xcache=False, af_name="", json_file='ntuples_merged.json', startind=0):
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
    with open(json_file) as f:
        file_info = json.load(f)

    # process into "fileset" summarizing all info
    fileset = {}
    for process in file_info.keys():
        # print("process = ", process)
        if process == "data":
            continue  # skip data

        for variation in file_info[process].keys():
            # print("variation = ", variation)
            file_list = file_info[process][variation]["files"]
            if n_files_max_per_sample != -1:
                if len(file_list)>0:
                    file_list = file_list[startind:n_files_max_per_sample+startind]  # use partial set of samples

            file_paths = [f["path"] for f in file_list]
            if use_xcache:
                file_paths = [f.replace("https://xrootd-local.unl.edu:1094", "root://red-xcache1.unl.edu") for f in file_paths]
            if af_name == "ssl-dev":
                # point to local files on /data
                file_paths = [f.replace("https://xrootd-local.unl.edu:1094//store/user/", "/data/alheld/") for f in file_paths]
            if len(file_list)>0:
                nevts_total = sum([f["nevts"] for f in file_list])
            else: nevts_total = 0
            metadata = {"process": process, "variation": variation, "nevts": nevts_total, "xsec": xsec_info[process]}
            fileset.update({f"{process}__{variation}": {"files": file_paths, "metadata": metadata}})

    return fileset


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


def make_datasource(fileset:dict, name: str, query: ObjectStream, ignore_cache: bool):
    """Creates a ServiceX datasource for a particular Open Data file."""
    datasets = [ServiceXDataset(fileset[name]["files"], backend_name="uproot", ignore_cache=ignore_cache)]
    return servicex.DataSource(
        query=query, metadata=fileset[name]["metadata"], datasets=datasets
    )


async def produce_all_histograms(fileset, query, analysis_processor, use_dask=False, ignore_cache=False, schema=None):
    """Runs the histogram production, processing input files with ServiceX and
    producing histograms with coffea.
    """
    # create the query
    ds = ServiceXSourceUpROOT("cernopendata://dummy", "events", backend_name="uproot")
    ds.return_qastle = True
    data_query = query(ds)

    # executor: local or Dask (Dask is not supported yet)
    if not use_dask:
        executor = servicex.LocalExecutor()
    else:
        executor = servicex.DaskExecutor(client_addr="tls://localhost:8786")

    datasources = [
        make_datasource(fileset, ds_name, data_query, ignore_cache=ignore_cache)
        for ds_name in fileset.keys()
    ]

    async def run_updates_stream(accumulator_stream, name):
        """Run to get the last item in the stream"""
        coffea_info = None
        try:
            async for coffea_info in accumulator_stream:
                pass
        except Exception as e:
            raise Exception(f"Failure while processing {name}") from e
        return coffea_info

    all_histogram_dicts = await asyncio.gather(
        *[
            run_updates_stream(
                executor.execute(analysis_processor, source, title=f"{source.metadata['process']}__{source.metadata['variation']}", schema=schema),
                f"{source.metadata['process']}__{source.metadata['variation']}",
            )
            for source in datasources
        ]
    )
    all_histograms = sum([h["hist"] for h in all_histogram_dicts])

    return all_histograms


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


def get_permutations_dict(MAX_N_JETS, include_labels=False):
    
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
        
    if include_labels:
        return permutations_dict, labels_dict
    else:
        return permutations_dict