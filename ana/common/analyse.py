import uproot
import awkward as ak
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hist
import gc
import sys 

from pyutils.pyselect import Select
from pyutils.pyvector import Vector
from pyutils.pylogger import Logger
from pyutils.pyprint import Print
from cut_manager import CutManager

# FIXME: doing a lot of copy operations is not very efficient, but I am not sure if there is an alternative

class Analyse:
    """Class to handle analysis functions
    """
    def __init__(self, on_spill=False, event_subrun=None, verbosity=1):
        """Initialise the analysis handler
        
        Args:
            on_spill (bool, optional): Include on-spill cuts
            event_subrun (tuple of ints, optional): Select a specific event and subrun
            verbosity (int, optional): Level of output detail (0: critical errors only, 1: info, 2: debug, 3: deep debug)
        """

        # Verbosity 
        self.verbosity = verbosity
        
        # Start logger
        self.logger = Logger(
            print_prefix="[Analyse]",
            verbosity=self.verbosity
        )
        
        # Initialise tools
        self.selector = Select(verbosity=self.verbosity)
        self.vector = Vector(verbosity=self.verbosity) 
        
        # Analysis configuration
        self.on_spill = on_spill  # Default to off-spill 
        self.event_subrun = event_subrun # event selection (for debugging)
        self.logger.log(f"Initialised with on_spill={self.on_spill}", "info")
    
    def define_cuts(self, data, cut_manager, on_spill=None):
        """Define analysis cuts

        Note that all cuts here need to be defined at trk level. 

        Also note that the tracking algorthm produces cut for upstream/downstream muon/electrons and then uses trkqual to guess the right one
        trkqual needs to be good before making a selection 
        this is particulary important for the pileup cut, since it needs to be selected from tracks which are above 90% or whatever 
        
        Args:
            data (ak.Array): data to apply cuts to
            cut_manager: The CutManager instance to use
            on_spill (bool, optional): Whether to apply on-spill specific cuts
        """
        
        if on_spill is None:
            on_spill = self.on_spill

        self.logger.log(f"Defining cuts (on_spill={on_spill})", "info")
            
        selector = self.selector

        # Track segments cuts
        try:
            
            # Tracker surfaces
            at_trk_front = selector.select_surface(data["trkfit"], sid=0) #
            at_trk_mid = selector.select_surface(data["trkfit"], sid=1)
            at_trk_back = selector.select_surface(data["trkfit"], sid=2)
            in_trk = (at_trk_front | at_trk_mid | at_trk_back)

            # Append: this is useful for plotting and debugging
            data["at_trk_front"] = at_trk_front

            # 1. Electron tracks 
            
            # Truth track parent is electron 
            is_electron = data["trkmc"]["trkmcsim"]["pdg"] == 11
            is_trk_parent = data["trkmc"]["trkmcsim"]["nhits"] == ak.max(data["trkmc"]["trkmcsim"]["nhits"], axis=-1)
            is_trk_parent_electron = is_electron & is_trk_parent 
            has_trk_parent_electron = ak.any(is_trk_parent_electron, axis=-1) # Any tracks with electron parents?
        
            cut_manager.add_cut(
                name="is_truth_electron", 
                description="Track parents are electrons (truth PID)", 
                mask=has_trk_parent_electron 
            )

            # Append track-level definition
            data["is_truth_electron"] = has_trk_parent_electron

            # Reco track fit is electron 
            is_reco_electron = selector.is_electron(data["trk"])
            data["is_reco_electron"] = is_reco_electron
        
            cut_manager.add_cut(
                name="is_reco_electron", 
                description="Tracks are assumed to be electrons (trk)", 
                mask=is_reco_electron 
            )

            # Append track-level definition
            data["is_reco_electron"] = is_reco_electron

            # 2. One reconstructed electron track / event 
            # Regardless of quality

            # Track-level definition
            one_reco_electron = ak.sum(is_reco_electron, axis=-1) == 1
            # Broadcast to track level
            one_reco_electron_per_event, _ = ak.broadcast_arrays(one_reco_electron, is_reco_electron) # this returns a tuple
            
            cut_manager.add_cut(
                name="one_reco_electron",
                description="One reco electron / event",
                mask=one_reco_electron_per_event 
            )

            # Append track-level definition
            data["one_reco_electron"] = one_reco_electron

            # Append event-level definition
            data["one_reco_electron_per_event"] = one_reco_electron_per_event

            # 3. Track fit quality
            # All tracks must have a fit quality of better than 80% or better
            # good_trkqual = (data["trk"]["trkqual.result"] > 0.8)
            good_trkqual = selector.select_trkqual(data["trk"], quality=0.8)
            cut_manager.add_cut(
                name="good_trkqual",
                description="Track quality (quality > 0.8)",
                mask=good_trkqual 
            )
            data["good_trkqual"] = good_trkqual

            # 4. Downstream tracks only through tracker entrance 
            self.logger.log("Defining downstream tracks cut", "max")
            # is_downstream = (data["trkfit"]["trksegs"]["mom"]["fCoordinates"]["fZ"] > 0)
            is_downstream = selector.is_downstream(data["trkfit"]) # at tracker entrance
            has_downstream = ak.any(is_downstream, axis=-1)
            
            cut_manager.add_cut(
                name="downstream",
                description="Downstream tracks (p_z > 0 through tracker)",
                mask=has_downstream 
            )

            # trksegs-level definition
            data["is_downstream"] = is_downstream
            # trk-level definition
            data["has_downstream"] = has_downstream

            # No reflections through tracker entrance
            # This one doesn't really work
            # self.logger.log("Defining reflection cut", "max")
            # is_reflected = selector.is_reflected(data["trkfit"])
            
            # cut_manager.add_cut(
            #     name="not_reflected",
            #     description="No reflected tracks",
            #     mask=~is_reflected 
            # )

            # # trk definition
            # data["is_reflected"] = is_reflected
            
            # One quality electron track / event 
            # We definitely do not want this one
            # It excludes potential signals if you happen to have a good muon fit to an electron track
            # self.logger.log("Defining pileup cut", "max")
            # no_pileup = ak.sum(good_trkqual & is_reco_electron, axis=-1) == 1 
            # # Broadcast to track level
            # no_pileup, _ = ak.broadcast_arrays(no_pileup, good_trkqual) # this returns a tuple

            # cut_manager.add_cut(
            #     name="no_pileup",
            #     description="One quality e- track / event",
            #     mask=no_pileup 
            # )
            
            # Append for debugging
            # data["no_pileup"] = no_pileup
            
            # 3. Minimum hits
            has_hits = selector.has_n_hits(data["trk"], n_hits=20)
            cut_manager.add_cut(
                name="has_hits",
                description="Minimum of 20 active hits in the tracker",
                mask=has_hits 
            )
        
            if on_spill:
                # 4. Time at tracker entrance (trk level)
                self.logger.log("Defining time cut (on-spill specific)", "info")

                # trksegs level
                within_t0 = ((640 < data["trkfit"]["trksegs"]["time"]) & 
                             (data["trkfit"]["trksegs"]["time"] < 1650))
            
                # trk-level definition (the actual cut)
                within_t0 = ak.all(~at_trk_front | within_t0, axis=-1)
                cut_manager.add_cut( 
                    name="within_t0",
                    description="t0 at tracker entrance (640 < t_0 < 1650 ns)",
                    mask=within_t0 
                )
                
            # 6. Loop helix maximum radius
            within_lhr_max = ((450 < data["trkfit"]["trksegpars_lh"]["maxr"]) & 
                              (data["trkfit"]["trksegpars_lh"]["maxr"] < 680)) # changed from 650
        
            # trk-level definition (the actual cut)
            within_lhr_max = ak.all(~at_trk_front | within_lhr_max, axis=-1)
            cut_manager.add_cut(
                name="within_lhr_max",
                description="Loop helix maximum radius (450 < R_max < 680 mm)",
                mask=within_lhr_max
            )

            # 6.5. Loose loop helix maximum radius
            within_lhr_max_loose = (data["trkfit"]["trksegpars_lh"]["maxr"] < 680) 
        
            # trk-level definition (the actual cut) 
            within_lhr_max_loose = ak.all(~at_trk_front | within_lhr_max_loose, axis=-1)
            cut_manager.add_cut(
                name="within_lhr_max_loose",
                description="Loop helix maximum radius (R_max < 680 mm)",
                mask=within_lhr_max_loose,
                active=False # OFF by default
            )
            
            # 7. Distance from origin
            within_d0 = (data["trkfit"]["trksegpars_lh"]["d0"] < 100)
        
            # trk-level definition (the actual cut)
            within_d0 = ak.all(~at_trk_front | within_d0, axis=-1) 
            cut_manager.add_cut(
                name="within_d0",
                description="Distance of closest approach (d_0 < 100 mm)",
                mask=within_d0 
                
            )
            
            # 8. Pitch angle
            within_pitch_angle = ((0.5577350 < data["trkfit"]["trksegpars_lh"]["tanDip"]) & 
                                  (data["trkfit"]["trksegpars_lh"]["tanDip"] < 1.0))
        
            # trk-level definition (the actual cut) 
            within_pitch_angle = ak.all(~at_trk_front | within_pitch_angle, axis=-1)
            cut_manager.add_cut(
                name="within_pitch_angle",
                description="Extrapolated pitch angle (0.5577350 < tan(theta_Dip) < 1.0)",
                mask=within_pitch_angle
            )

            # 8.5. Loose pitch angle
            within_pitch_angle_loose = (data["trkfit"]["trksegpars_lh"]["tanDip"] < 2.0)
        
            # trk-level definition (the actual cut) 
            within_pitch_angle_loose = ak.all(~at_trk_front | within_pitch_angle_loose, axis=-1)
            cut_manager.add_cut(
                name="within_pitch_angle_loose",
                description="Extrapolated pitch angle (tan(theta_Dip) < 2.0)",
                mask=within_pitch_angle_loose,
                active=False # OFF by default
            )

            # 9. CRV veto: |dt| < 150 ns (dt = coinc time - track t0) 
            # Check if EACH track is within 150 ns of ANY coincidence 
            
            dt_threshold = 150
            
            # Get track and coincidence times
            trk_times = data["trkfit"]["trksegs"]["time"][at_trk_front]  # events × tracks × segments
            coinc_times = data["crv"]["crvcoincs.time"]                  # events × coincidences
            
            # Broadcast CRV times to match track structure, so that we can compare element-wise
            # FIXME: should use ak.broadcast
            coinc_broadcast = coinc_times[:, None, None, :]  # Add dimensions for tracks and segments
            trk_broadcast = trk_times[:, :, :, None]         # Add dimension for coincidences

            # coinc_broadcast shape is [E, 1, 1, C] 
            # trk_broadcast shape is [E, T, S, 1]
            
            # Calculate time differences
            dt = abs(trk_broadcast - coinc_broadcast)
            
            # Check if within threshold
            within_threshold = dt < dt_threshold

            # For plotting: find the coincidences within threshold 
            # coinc_veto = ak.any(within_threshold, axis=(1, 2))  # [E, C]
            # data["coinc_veto"] = coinc_veto
            
            # Reduce one axis at a time 
            # First reduce over coincidences (axis=3)
            any_coinc = ak.any(within_threshold, axis=3)
            
            # Then reduce over trks (axis=2) 
            veto = ak.any(any_coinc, axis=2)

            data["unvetoed"] = ~veto

            cut_manager.add_cut(
                name="unvetoed",
                description="No veto: |dt| >= 150 ns",
                mask=~veto
            )
            
            self.logger.log("All cuts defined", "success")
            
        except Exception as e:
            self.logger.log(f"Error defining cuts: {e}", "error") 
            return None  
        
    def apply_cuts(self, data, cut_manager, group=None, active_only=True):

        ## data_cut needs to be an awkward array 
    
        """Apply all trk-level mask to the data
        
        Args:
            data: Data to apply cuts to
            mask: Mask to apply 
            
        Returns:
            ak.Array: Data after cuts applied
        """
        self.logger.log("Applying cuts to data", "info")
        
        try:
            # Copy the array 
            # This is memory intensive but the easiest solution for what I'm trying to do
            data_cut = ak.copy(data) 
            
            # Combine cuts
            self.logger.log(f"Combining cuts", "info") 

            # Track-level mask
            trk_mask = cut_manager.combine_cuts(active_only=active_only)
            
            # Select tracks
            self.logger.log("Selecting tracks", "max")
            data_cut["trk"] = data_cut["trk"][trk_mask]
            data_cut["trkfit"] = data_cut["trkfit"][trk_mask]
            data_cut["trkmc"] = data_cut["trkmc"][trk_mask]

            # OPTIONAL: select track segments in tracker 
            # This is handled by 
            # This makes it easier to study background events in the end
            # This seems to make a massive difference 
            # data_cut["trkfit"] = data_cut["trkfit"][data_cut["at_trk_front"]]
            
            # Then clean up events with no tracks after cuts
            self.logger.log(f"Cleaning up events with no tracks after cuts", "max") 
            data_cut = data_cut[ak.any(trk_mask, axis=-1)] 
            
            self.logger.log(f"Cuts applied successfully", "success")
            
            return data_cut
            
        except Exception as e:
            self.logger.log(f"Error applying cuts: {e}", "error") 
            return None
            
    def create_histograms(self, data, data_CE, data_CE_unvetoed):
        
        """Create histograms from the data before and after applying cuts
        
        Args:
            data: Data before cuts
            data_cut: Data after cuts
            
        Returns:
            dict: Dictionary of histograms
        """
        self.logger.log("Creating histograms", "info")
        
        # Tools 
        selector = self.selector 
        vector = self.vector

        hist_labels = ["All", "CE-like", "Unvetoed CE-like"]

        try: 

            #### Create histogram objects:
            # Full momentum range histogram
            h1_mom_full_range = hist.Hist(
                hist.axis.Regular(30, 0, 300, name="momentum", label="Momentum [MeV/c]"),
                hist.axis.StrCategory(hist_labels, name="selection", label="Selection")
            )
            # Signal region histogram (fine binning)
            h1_mom_signal_region = hist.Hist(
                hist.axis.Regular(13, 103.6, 104.9, name="momentum", label="Momentum [MeV/c]"),
                hist.axis.StrCategory(hist_labels, name="selection", label="Selection")
            )

            # Z-position histograms
            h1_crv_z = hist.Hist(
                hist.axis.Regular(100, -15e3, 10e3, name="crv_z", label="CRV z-position [mm]"),
                hist.axis.StrCategory(hist_labels, name="selection", label="Selection")
            )


            # Process and fill histograms in batches
            def _fill_hist(data, label): 
                """ Nested helper function to fill hists """

                # Tracks must be electron candidates at the tracker entrance
                is_electron = selector.is_electron(data["trk"])  
                data["trkfit"] = data["trkfit"][is_electron]
                at_trk_front = selector.select_surface(data["trkfit"], sid=0)              
                mom = vector.get_mag(data["trkfit"]["trksegs"][at_trk_front], "mom")
                crv_z = ak.flatten(data["crv"]["crvcoincs.pos.fCoordinates.fZ"], axis=None)
                
                # Flatten 
                if mom is None:
                    mom = ak.Array([])
                else:
                    mom = ak.flatten(mom, axis=None)
                    
                # Fill h1ogram for wide range
                h1_mom_full_range.fill(momentum=mom, selection=np.full(len(mom), label))

                # Fill signal region
                mom_sig = mom[(mom >= 103.6) & (mom <= 104.9)]
                h1_mom_signal_region.fill(momentum=mom_sig, selection=np.full(len(mom_sig), label))

                # Fill position
                h1_crv_z.fill(crv_z=crv_z, selection=np.full(len(crv_z), label))
                
                # Clean up 
                del mom, mom_sig, crv_z
                import gc
                gc.collect()
                    
            # All need to be at tracker entrance
            # 1. First process "all tracks" data
            _fill_hist(data, "All") 
            _fill_hist(data_CE, "CE-like")
            _fill_hist(data_CE_unvetoed, "Unvetoed CE-like")
    
            self.logger.log("Histograms filled", "success")

            # Create a copy in results and explicitly delete large arrays after use
            result = {
                "Wide range": h1_mom_full_range.copy(), 
                "Signal region": h1_mom_signal_region.copy(),
                "CRV z-position": h1_crv_z.copy()
            }

            return result 

        except Exception as e:
            # Handle any errors that occur during processing
            self.logger.log(f"Error filling histograms: {e}", "error")
            return None
        
    def execute(self, data, file_id, cuts_to_toggle=None):
        """Perform complete analysis on an array
        
        Args:
            data: The data to analyse
            file_id: Identifier for the file
            cuts_to_toggle: Dict of cut name with active state  
            
        Returns:
            dict: Complete analysis results
        """

        self.logger.log(f"Beginning analysis execution for file: {file_id}", "info") 
        
        try:

            # Create a unique cut manager for this file
            cut_manager = CutManager(verbosity=self.verbosity)

            # Optional prefiltering 
            if self.event_subrun is not None: 
                mask = ((data["evt"]["event"] == self.event_subrun[0]) & (data["evt"]["subrun"] == self.event_subrun[1]))
                data = data[mask]

            # Define cuts
            self.logger.log("Defining cuts", "max")
            self.define_cuts(data, cut_manager)

            # Set activate cuts
            if cuts_to_toggle: 
                cut_manager.toggle_cut(cuts_to_toggle) 
            
            # Calculate cut stats
            self.logger.log("Getting cut stats", "max")
            cut_stats = cut_manager.calculate_cut_stats(data, progressive=True, active_only=True)

            # Apply CE-like cuts
            self.logger.log("Applying cuts", "max")

            # Turn off veto 
            cut_manager.toggle_cut({"unvetoed" : False})
            # Mark CE-like tracks (useful for debugging 
            data["CE_like"] = cut_manager.combine_cuts(active_only=True)
            # Apply cuts
            data_CE = self.apply_cuts(data, cut_manager) # Just CE-like tracks 
            
            # Turn on veto 
            data_CE_unvetoed = None
            cut_manager.toggle_cut({"unvetoed" : True})
            # Mark CE-like tracks (useful for debugging 
            data["unvetoed_CE_like"] = cut_manager.combine_cuts(active_only=True)
            # Apply cuts
            data_CE_unvetoed = self.apply_cuts(data, cut_manager) 
        
            
            # Create histograms
            self.logger.log("Creating histograms", "max")
            histograms = self.create_histograms(data, data_CE, data_CE_unvetoed)
            
            # Compile all results
            self.logger.log("Analysis completed", "success")

            result = {
                "file_id": file_id,
                "cut_stats": cut_stats,
                "filtered_data": data_CE_unvetoed, 
                "histograms": histograms
            }

            # Force garbage collection
            gc.collect()
            
            return result
            
        except Exception as e:
            self.logger.log(f"Error during analysis execution: {e}", "error")  
            return None

class Utils():
    """
    Utils class for misc helper methods 

    These do not fit into the main analysis process
    """
    def __init__(self, verbosity=1):
        """Initialise
        """
        # Verbosity 
        self.verbosity = verbosity
        # Start logger
        self.logger = Logger(
            print_prefix="[PostProcess]",
            verbosity=self.verbosity
        )
        # Selector 
        self.selector = Select(verbosity=0)
        # Printer
        self.printer = Print(verbose=True)
        # Confirm
        self.logger.log(f"Initialised", "info")

    def get_background_events(self, results, printout=True, out_path=None): 
        """
        Write background event info

        Args: 
            results (list): list of results 
            out_path: File path for txt output 
        """
        output = []
        count = 0
        
        for i, result in enumerate(results): 
            
            data = ak.Array(result["filtered_data"])
            
            if len(data) == 0:
                continue

            # Get tracker entrance times
            trk_front = self.selector.select_surface(data["trkfit"], sid=0)
            track_time = data["trkfit"]["trksegs"]["time"][trk_front]
            # Get coinc entrance times
            coinc_time = data["crv"]["crvcoincs.time"]
            
            # Extract values
            track_time_str = "" 
            coinc_time_str = ""
            
            # Extract floats from track_time (nested structure: [[[values]], [[values]]])
            track_floats = []
            for nested in track_time:
                for sublist in nested:
                    for val in sublist:
                        track_floats.append(float(val))
            
            # Extract floats from coinc_time (structure: [[], []])
            coinc_floats = []
            for sublist in coinc_time:
                for val in sublist:
                    coinc_floats.append(float(val))
            
            # Format as strings with precision
            if track_floats:
                track_time_str = ", ".join([f"{val:.6f}" for val in track_floats])
            
            if coinc_floats:
                coinc_time_str = ", ".join([f"{val:.6f}" for val in coinc_floats])
        
            # Calculate dt
            dt_str = ""
            if track_floats and coinc_floats:
                # Calculate dt between first track time and first coinc time
                dt_value = abs(track_floats[0] - coinc_floats[0])
                dt_str = f"{dt_value:.6f}"
            
            output.append(f"  Index:            {i}")
            output.append(f"  Subrun:           {data["evt"]["subrun"]}")
            output.append(f"  Event:            {data["evt"]["event"]}")
            output.append(f"  File:             {result["file_id"]}")
            output.append(f"  Track time [ns]:  {track_time_str}") 
            output.append(f"  Coinc time [ns]:  {coinc_time_str if len(coinc_time_str)>0 else None}") 
            output.append(f"  dt [ns]:          {dt_str if len(dt_str)>0 else "N/A"}")
            output.append("-" * 40)

            count += 1
        
        output = "\n".join(output)
        
        # Print 
        if printout:
            self.logger.log(f"Info for {count} background events :", "info")
            print(output)
        
        # Write to file
        if out_path:
            with open(out_path, "w") as f:
                f.write(output)
        
            self.logger.log(f"Wrote {out_path}", "success")

    def get_verbose_background_events(self, data, out_path):

        # Redirect stdout to file
        with open(out_path, "w") as f:
            old_stdout = sys.stdout
            sys.stdout = f
            self.printer.print_n_events(data, n_events=len(data))
            # Restore stdout
            sys.stdout = old_stdout
            self.logger.log(f"Wrote {out_path}", "success")
    

    def get_kN(self, df_stats, numerator_name=None, denominator_name=None):
        """
        Retrieve efficiency data from DataFrame
        
        Args:
            df_stats (pandas.DataFrame): DataFrame with cut statistics
            numerator_name: The row to use as numerator.
            denominator_name: The row to use as denominator_name.
            
        Returns:
            tuple: k, N
        """
    
        if numerator_name is None or denominator_name is None:
            logger.log("Please provide a numerator_name and a denominator_name", "error")
            return None
    
        # Get numerator
        numerator_row = df_stats[df_stats["Cut"] == numerator_name]
        k = numerator_row["Events Passing"].iloc[0]
    
        # Get denominator
        denominator_row = df_stats[df_stats["Cut"] == denominator_name]
        N = denominator_row["Events Passing"].iloc[0]
    
        # Hard to do this and handle all edge cases
        # # Get numerator
        # k = None
        # if numerator_name: # Use specified numerator
        # numerator_row = df_stats[df_stats["Cut"] == numerator_name]
        # k = numerator_row["Events Passing"].iloc[0]
        # else: 
        #     if not signal: # Use last row
        #         last_row = df_stats.iloc[-1]
        #         k = last_row["Events Passing"]
        #     else: # Use penultimate row
        #         penultimate_row = df_stats.iloc[-2]
        #         k = penultimate_row["Events Passing"]
                
        # # Get denominator
        # N = None
        # if denominator_name:  # Use specified denominator
        #     denominator_row = df_stats[df_stats["Cut"] == denominator_name]
        #     N = denominator_row["Events Passing"].iloc[0]
        # else: 
        #     if not signal: # Use penultimate row
        #         penultimate_row = df_stats.iloc[-2]
        #         N = penultimate_row["Events Passing"]
        #     else: # Use first row
        #         first_row = df_stats.iloc[0]
        #         N = first_row["Events Passing"]
        
        return k, N
    
    def get_eff(self, df_stats, ce_row_name="within_pitch_angle", veto=True):
        
        """
        Report efficiency results
        """

        self.logger.log(f"Getting efficiency with ce_row_name = {ce_row_name} and veto = {veto}", "info")
    
        # 
        results = []
        
        k_sig, N_sig = self.get_kN(
            df_stats, 
            numerator_name = ce_row_name, 
            denominator_name = "No cuts"
        ) 
    
        # Calculate efficiency
        eff_sig = (k_sig / N_sig) if N_sig > 0 else 0
        # Calculate poisson uncertainty 
        eff_sig_err = np.sqrt(k_sig) / N_sig
    
        results.append({
            "Type": "Signal",
            "Events Passing (k)": k_sig,
            "Total Events (N)": N_sig,
            "Efficiency [%]": eff_sig * 100,
            "Efficiency Error [%]": eff_sig_err * 100
        })
    
        if veto:
        
                k_veto, N_veto = None, None 
        
                k_veto, N_veto = self.get_kN(
                    df_stats, 
                    numerator_name = "unvetoed", 
                    denominator_name = ce_row_name
                ) 
        
                # Calculate efficiency
                eff_veto = 1 - (k_veto / N_veto) if N_veto > 0 else 0
                # Calculate poisson uncertainty 
                eff_veto_err = np.sqrt(k_veto) / N_veto
        
                results.append({
                    "Type": "Veto",
                    "Events Passing (k)": k_veto,
                    "Total Events (N)": N_veto,
                    "Efficiency [%]": eff_veto * 100,
                    "Efficiency Error [%]": eff_veto_err * 100
                })

        self.logger.log(f"Returning efficiency information", "success")
        
        return pd.DataFrame(results)


    
    # def get_background_events(self, results, printout=True, out_path=None): 
    #     """
    #     Write background event info

    #     Args: 
    #         results (list): list of results 
    #         out_path: File path for txt output 
    #     """
    #     output = []
    #     count = 0
        
    #     for i, result in enumerate(results): 
            
    #         data = ak.Array(result["filtered_data"])
            
    #         if len(data) == 0:
    #             continue

    #         # Get tracker entrance times
    #         trk_front = self.selector.select_surface(data["trkfit"], sid=0)
    #         track_time = data["trkfit"]["trksegs"]["time"][trk_front]
    #         # Get coinc entrance times
    #         coinc_time = data["crv"]["crvcoincs.time"]
            
    #         # Extract values
    #         track_time_str = "" 
    #         coinc_time_str = ""
            
    #         # Extract floats from track_time (nested structure: [[[values]], [[values]]])
    #         track_floats = []
    #         for nested in track_time:
    #             for sublist in nested:
    #                 for val in sublist:
    #                     track_floats.append(float(val))
            
    #         # Extract floats from coinc_time (structure: [[], []])
    #         coinc_floats = []
    #         for sublist in coinc_time:
    #             for val in sublist:
    #                 coinc_floats.append(float(val))
            
    #         # Format as strings with precision
    #         if track_floats:
    #             track_time_str = ", ".join([f"{val:.6f}" for val in track_floats])
            
    #         if coinc_floats:
    #             coinc_time_str = ", ".join([f"{val:.6f}" for val in coinc_floats])
        
    #         # Calculate dt
    #         dt_str = ""
    #         if track_floats and coinc_floats:
    #             # Calculate dt between first track time and first coinc time
    #             dt_value = abs(track_floats[0] - coinc_floats[0])
    #             dt_str = f"{dt_value:.6f}"
            
    #         output.append(f"  Index:            {i}")
    #         output.append(f"  Subrun:           {data["evt"]["subrun"]}")
    #         output.append(f"  Event:            {data["evt"]["event"]}")
    #         output.append(f"  File:             {result["file_id"]}")
    #         output.append(f"  Track time [ns]:  {track_time_str}") 
    #         output.append(f"  Coinc time [ns]:  {coinc_time_str if len(coinc_time_str)>0 else None}") 
    #         output.append(f"  dt [ns]:          {dt_str if len(dt_str)>0 else "N/A"}")
    #         output.append("-" * 40)

    #         count += 1
        
    #     output = "\n".join(output)
        
    #     # Print 
    #     if printout:
    #         self.logger.log(f"Info for {count} background events :", "info")
    #         print(output)
        
    #     # Write to file
    #     if out_path:
    #         with open(fout_name, "w") as f:
    #             f.write(output)
        
    #         self.logger.log(f"\tWrote {out_path}", "success")
            
    # # This could be quite nice, but it needs the full analysis config including everything which could be a bit complex. 
    # # Need to think about this. 
            
    # def save_configuration(self, filename):
    #     """Save the current analysis configuration to a file
        
    #     Args:
    #         filename (str): Path to the file to save the configuration to
    #     """
    #     self._log(f"Saving configuration to {filename}", level=1)
        
    #     try:
            
    #         # Create a configuration dictionary
    #         config = {
    #             "verbosity": self.verbosity,
    #             "on_spill": self.on_spill,
    #             # Add other configuration parameters as needed
    #         }
            
    #         # Save to file
    #         with open(filename, 'w') as f:
    #             json.dump(config, f, indent=4)
                
    #         self._log("Configuration saved successfully", level=1)
            
    #     except Exception as e:
    #         self._log(f"Error saving configuration: {e}", level=0)
    #         raise
            
    # def load_configuration(self, filename):
    #     """Load an analysis configuration from a file
        
    #     Args:
    #         filename (str): Path to the configuration file
    #     """
    #     self._log(f"Loading configuration from {filename}", level=1)
        
    #     try:
    #         import json
            
    #         # Load from file
    #         with open(filename, 'r') as f:
    #             config = json.load(f)
                
    #         # Apply configuration
    #         self.verbosity = config.get("verbosity", self.verbosity)
    #         self.on_spill = config.get("on_spill", self.on_spill)
    #         # Apply other configuration parameters as needed
            
    #         # Update dependent objects
    #         if hasattr(self, 'selector'):
    #             self.selector.verbosity = self.verbosity
    #         if hasattr(self, 'vector'):
    #             self.vector.verbosity = self.verbosity
                
    #         self._log("Configuration loaded successfully", level=1)
            
    #     except Exception as e:
    #         self._log(f"Error loading configuration: {e}", level=0)
            # raise