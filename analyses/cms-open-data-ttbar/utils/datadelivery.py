import hist
import json
import numpy as np
from servicex import ServiceXDataset


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


def construct_fileset(n_files_max_per_sample, use_xcache=False, af_name=""):
    # using https://atlas-groupdata.web.cern.ch/atlas-groupdata/dev/AnalysisTop/TopDataPreparation/XSection-MC15-13TeV.data
    # for reference
    # x-secs are in pb
    xsec_info = {
        "ttbar": 396.87 + 332.97,  # nonallhad + allhad, keep same x-sec for all
        "single_top_s_chan": 2.0268 + 1.2676,
        "single_top_t_chan": (36.993 + 22.175)
        / 0.252,  # scale from lepton filter to inclusive
        "single_top_tW": 37.936 + 37.906,
        "wjets": 61457 * 0.252,  # e/mu+nu final states
        "data": None,
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
                file_list = file_list[
                    :n_files_max_per_sample
                ]  # use partial set of samples

            file_paths = [f["path"] for f in file_list]
            if use_xcache:
                file_paths = [
                    f.replace(
                        "https://xrootd-local.unl.edu:1094",
                        "root://red-xcache1.unl.edu",
                    )
                    for f in file_paths
                ]
            if af_name == "ssl-dev":
                # point to local files on /data
                file_paths = [
                    f.replace(
                        "https://xrootd-local.unl.edu:1094//store/user/",
                        "/data/alheld/",
                    )
                    for f in file_paths
                ]
            nevts_total = sum([f["nevts"] for f in file_list])
            metadata = {
                "process": process,
                "variation": variation,
                "nevts": nevts_total,
                "xsec": xsec_info[process],
            }
            fileset.update(
                {f"{process}__{variation}": {"files": file_paths, "metadata": metadata}}
            )

    return fileset


class ServiceXDatasetGroup:
    def __init__(self, fileset, backend_name="uproot", ignore_cache=False):
        self.fileset = fileset

        # create list of files (& associated processes)
        filelist = []
        for i, process in enumerate(fileset):
            filelist += [[filename, process] for filename in fileset[process]["files"]]

        filelist = np.array(filelist)
        self.filelist = filelist
        self.ds = ServiceXDataset(
            filelist[:, 0].tolist(),
            backend_name=backend_name,
            ignore_cache=ignore_cache,
        )

    def get_data_rootfiles_uri(self, query, as_signed_url=True, title="Untitled"):
        all_files = np.array(
            self.ds.get_data_rootfiles_uri(
                query, as_signed_url=as_signed_url, title=title
            )
        )
        parent_file_urls = np.array([f.file for f in all_files])

        # order is not retained after transform, so we can match files to their parent files using the filename
        # (replacing / with : to mitigate servicex filename convention )
        parent_key = np.array(
            [
                np.where(parent_file_urls == self.filelist[i][0].replace("/", ":"))[0][
                    0
                ]
                for i in range(len(self.filelist))
            ]
        )

        files_per_process = {}
        for i, process in enumerate(self.fileset):
            # update files for each process
            files_per_process.update(
                {process: all_files[parent_key[self.filelist[:, 1] == process]]}
            )

        return files_per_process


def create_hist_dict(channel_names, num_bins, bin_low, bin_high, hist_label):
    """
    Initialize dictionary of histograms accummulated in coffea processor

    Args:
        channel_names: arraylike object containing channel names, used as keys of dictionary
        num_bins: arraylike (same length as channel_names) or scalar specifying number of bins per histogram
        bin_low: arraylike (same length as channel_names) object or scalar specifying lower histogram ranges.
        bin_high: arraylike (same length as channel_names) object or scalar specifying upper histogram ranges.
        hist_label: arraylike (same length as channel_names) or scalar specifying label of histogram (displays while plotting)

    Returns:
        hist_dict: dictionary containing histograms associated with each channel
    """
    # ensure arguments have correct length
    n_channels = len(channel_names)
    for arg in [num_bins, bin_low, bin_high, hist_label]:
        if hasattr(arg, "__len__"):
            if not len(arg)==n_channels:
                raise ValueError(
                    f"If {arg} is not a scalar, it must have length equal to that of channel_names"
                )
                
    hist_dict = {}
    for i in range(len(channel_names)):
        hist_args = [0,0,0,0]
        for j, arg in enumerate([num_bins, bin_low, bin_high, hist_label]):
            if hasattr(arg, "__len__"):
                hist_args[j] = arg[i]
            else:
                hist_args[j] = arg
        
        hist_dict[channel_names[i]] = (
            hist.Hist.new.Reg(
                hist_args[0], # num_bins
                hist_args[1], # bin_low
                hist_args[2], # bin_high
                name="observable",
                label=hist_args[3], # hist_label
            )
            .StrCat([], name="process", label="Process", growth=True)
            .StrCat([], name="variation", label="Systematic variation", growth=True)
            .Weight()
        )

    return hist_dict
