# mu2e_cosmic_ana

## Description

This repository contains code and configuration files for performing **sensitivity studies of the Mu2e Cosmic Ray Veto (CRV) system** using both on-spill and off-spill datasets. The analysis framework allows users to explore the impact of different cut strategies on CRV performance, including customizable selection criteria and data subsets.

The tools in this repository were originally developed by **Sam Grant** and extended by **Victor Dorojan** as part of ongoing efforts to optimize CRV signal efficiency and background rejection. The analysis aims to support detector studies by providing **reproducible workflows for cosmic background characterization**.

### Key Components in the `common/` Folder:

- **`analyse.py`**  
  Processes particle tracking data and applies selection cuts to identify electron tracks using both truth-level and reconstructed information. It sets up logging, selection utilities, and prepares the data for further analysis or plotting.

- **`cut_manager.py`**  
  Defines the `CutManager` class used to manage, apply, and analyze logical cuts on particle physics data. It allows cuts to be added, toggled on/off, combined logically, and used to produce detailed statistics on event selection.

- **`postprocess.py`**  
  Defines a `PostProcess` class that consolidates filtered data, histograms, and cut statistics from multiple analysis result files. It merges awkward arrays, combines histograms, and aggregates cut statistics using `CutManager`, streamlining the final analysis stage.

---

## Contents

1. `common/` – Core analysis utilities (`analyse.py`, `cut_manager.py`, `postprocess.py`)
2. `comp/` – Comparison tools for datasets or configurations
3. `models/` – Models and configuration files used in the analysis
4. `offspill/` – Off-spill data analysis studies
5. `onspill/` – On-spill data analysis studies
6. `signal/` – Signal characterization, including efficiency and background studies

---

## Users Can:

- Toggle cut parameters to suit their own analysis goals  
- Reproduce and extend results from existing studies (`offspill/` and `onspill/`)  
- Analyze efficiency, dead time, and background rates across datasets (`signal/`)  

---

This toolkit is designed to be **modular** and **user-friendly** for collaborators working on Mu2e cosmic background mitigation.
