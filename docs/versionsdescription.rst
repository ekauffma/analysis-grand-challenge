This AGC Version (v0.2)
================================

Datasets
-----------------
The datasets used for this AGC version are from the 2015 CMS Open Data release and are in miniAOD format. The list of datasets separated by process is included below:

* **ttbar**:

  * nominal:
    
    * `19980 <https://opendata.cern.ch/record/19980>`_: Powheg + Pythia 8 (ext3), 2413 files, 3.4 TB -> converted
    * `19981 <https://opendata.cern.ch/record/19981>`_: Powheg + Pythia 8 (ext4), 4653 files, 6.4 TB -> converted
    
  * scale variation:
    
    * `19982 <https://opendata.cern.ch/record/19982>`_: same as below, unclear if overlap
    * `19983 <https://opendata.cern.ch/record/19983>`_: Powheg + Pythia 8 "scaledown" (ext3), 902 files, 1.4 TB -> converted
    * `19984 <https://opendata.cern.ch/record/19984>`_: same as below, unclear if overlap
    * `19985 <https://opendata.cern.ch/record/19985>`_: Powheg + Pythia 8 "scaleup" (ext3), 917 files, 1.3 TB -> converted
  
  * ME variation:
    
    * `19977 <https://opendata.cern.ch/record/19977>`_: same as below, unclear if overlap
    * `19978 <https://opendata.cern.ch/record/19978>`_: aMC@NLO + Pythia 8 (ext1), 438 files, 647 GB -> converted
  
  * PS variation:
    
    * `19999 <https://opendata.cern.ch/record/19999>`_: Powheg + Herwig++, 443 files, 810 GB -> converted

* **single top**:

  * s-channel:
    
    * `19394 <https://opendata.cern.ch/record/19394>`_: aMC@NLO + Pythia 8, 114 files, 76 GB -> converted
  
  * t-channel:
    
    * `19406 <https://opendata.cern.ch/record/19406>`_: Powheg + Pythia 8 (antitop), 935 files, 1.1 TB -> converted
    * `19408 <https://opendata.cern.ch/record/19408>`_: Powheg + Pythia 8 (top), 1571 files, 1.8 TB -> converted
  
  * tW:
    
    * nominal:
      
      * `19412 <https://opendata.cern.ch/record/19412>`_: Powheg + Pythia 8 (antitop), 27 files, 30 GB -> converted
      * `19419 <https://opendata.cern.ch/record/19419>`_: Powheg + Pythia 8 (top), 23 files, 30 GB -> converted
    
    * DS:
      
      * `19410 <https://opendata.cern.ch/record/19410>`_: Powheg + Pythia 8 DS (antitop), 13 files, 15 GB
      * `19417 <https://opendata.cern.ch/record/19417>`_: Powheg + Pythia 8 DS (top), 13 files, 14 GB
    
    * scale variations:
      
      * `19415 <https://opendata.cern.ch/record/19415>`_: Powheg + Pythia 8 "scaledown" (antitop), 11 files, 15 GB
      * `19422 <https://opendata.cern.ch/record/19422>`_: Powheg + Pythia 8 "scaledown" (top), 13 files, 15 GB
      * `19416 <https://opendata.cern.ch/record/19416>`_: Powheg + Pythia 8 "scaleup" (antitop), 12 files, 14 GB
      * `19423 <https://opendata.cern.ch/record/19423>`_: Powheg + Pythia 8 "scaleup" (top), 13 files, 14 GB

    * there are also larger `NoFullyHadronicDecays` samples: `19411 <https://opendata.cern.ch/record/19411>`_, `19418 <https://opendata.cern.ch/record/19418>`_
  
  * tZ / tWZ: potentially missing in inputs, not included in `/ST_*`

* **W+jets**:

  * nominal (with 1l filter):
  
    * `20546 <https://opendata.cern.ch/record/20546>`_: same as below, unclear if overlap
    * `20547 <https://opendata.cern.ch/record/20547>`_: aMC@NLO + Pythia 8 (ext2), 5601 files, 4.5 TB -> converted
    * `20548 <https://opendata.cern.ch/record/20548>`_: aMC@NLO + Pythia 8 (ext4), 4598 files, 3.8 TB -> converted

* **data**:

  * single muon:
  
    * `24119 <https://opendata.cern.ch/record/24119>`_: 1916 files, 1.4 TB -> converted
  
  * single electron:
    
    * `24120 <https://opendata.cern.ch/record/24120>`_: 2974 files, 2.6 TB -> converted
  
  * validated runs:
    
    * `24210 <https://opendata.cern.ch/record/24210>`_: single txt file
    
More information about datasets can be found in `analysis-grand-challenge/datasets/cms-open-data-2015/`.

Cross-section values
-----------------
The values used for the cross-section of each process are included in the table below. These values were obtained from `https://atlas-groupdata.web.cern.ch/atlas-groupdata/dev/AnalysisTop/TopDataPreparation/XSection-MC15-13TeV.data <https://atlas-groupdata.web.cern.ch/atlas-groupdata/dev/AnalysisTop/TopDataPreparation/XSection-MC15-13TeV.data>`_.

.. list-table:: Cross-section Values
   :widths: 25 25 50
   :header-rows: 1

   * - Process
     - Crosssection (pb)
     - Comments
   * - ttbar
     - 396.87 + 332.97
     - nonallhad + allhad, keep same x-sec for all
   * - single_top_s_chan
     - 2.0268 + 1.2676
     - 
   * - single_top_t_chan
     - (36.993 + 22.175)/0.252
     - scale from lepton filter to inclusive
   * - single_top_tW
     - 37.936 + 37.906
     - e/mu+nu final states


Cuts
-----------------

The cuts applied in this analysis are listed below:

* Leptons (electrons and muons) must have :math:`p_T>25` GeV
* Events must contain exactly one lepton
* Jets must have :math:`p_T>25` GeV
* Events must have at least four jets
* Jets are considered :math:`b`-tagged if they have a :math:`b`-tag score over `B_TAG_THRESHOLD=0.5`.
* Events must have at least one :math:`b`-tagged jet
* **4j1b Region**: Events must have exactly one :math:`b`-tagged jet
* **4j2b Region**: Events must have two or more :math:`b`-tagged jets