import uproot
import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import hist
import gc

from pyutils.pyselect import Select
from pyutils.pyvector import Vector
from pyutils.pylogger import Logger
from cut_manager import CutManager

# FIXME: doing a lot of copy operations is not very efficient, but I am not sure if there is an alternative

class Analyse:
    """Class to handle analysis functions
    """
    def __init__(self, verbosity=1):
        """Initialise the analysis handler
        
        Args:
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
        self.on_spill = False  # Default to off-spill 
        self.logger.log(f"Initialised with on_spill={self.on_spill}", "info")
    
    def define_cuts(self, data, cut_manager, on_spill=None):
        """Define analysis cuts
        
        Args:
            data (ak.Array): data to apply cuts to
            cut_manager: The CutManager instance to use
            on_spill (bool, optional): Whether to apply on-spill specific cuts
        """
        
        if on_spill is None:
            on_spill = self.on_spill

        self.logger.log(f"Defining cuts (on_spill={on_spill})", "info")
            
        selector = self.selector
        
        try:
            # Tracker surfaces
            trk_front = selector.select_surface(data["trkfit"], sid=0)
            trk_mid = selector.select_surface(data["trkfit"], sid=1)
            trk_back = selector.select_surface(data["trkfit"], sid=2)
            in_trk = (trk_front | trk_mid | trk_back)

            # 0. Track segments at tracker entrance
            # cut_manager.add_cut(
            #     name="at_trk_ent", 
            #     description="Track segments at tracker entrance", 
            #     mask=trk_front
            # )

            # Useful for debugging
            data["at_trk_front"] = trk_front
            
            # 1. Truth track parent is electron 
            is_electron = data["trkmc"]["trkmcsim"]["pdg"] == 11
            is_trk_parent = data["trkmc"]["trkmcsim"]["nhits"] == ak.max(data["trkmc"]["trkmcsim"]["nhits"], axis=-1)
            is_trk_parent_electron = is_electron & is_trk_parent 
            has_trk_parent_electron = ak.any(is_trk_parent_electron, axis=-1) # Any tracks with electron parents?
        
            cut_manager.add_cut(
                name="is_truth_electron", 
                description="Track parents are electrons (truth PID)", 
                mask=has_trk_parent_electron # has_trk_parent_electron
            )

            # 1.5 Also include trk.pdg == 11
            is_trk_electron = selector.is_electron(data["trk"])
        
            cut_manager.add_cut(
                name="is_reco_electron", 
                description="Tracks are assumed to be electrons (trk)", 
                mask=is_trk_electron # has_trk_parent_electron
            )
            
            # 2. Downstream tracks only through tracker
            self.logger.log("Defining downstream tracks cut", "max")
            is_downstream = selector.is_downstream(data["trkfit"])
        
             # trkseg-level definition, useful for plotting
            data["is_downstream_in_trk"] = is_downstream & trk_front 
        
            # trk-level definition (the actual cut)
            is_downstream = ak.all(~in_trk | is_downstream, axis=-1)
            cut_manager.add_cut(
                name="is_downstream",
                description="Downstream tracks only (p_z > 0 through tracker)",
                mask=is_downstream
            )
            
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
                within_t0 = ((640 < data["trkfit"]["trksegs"]["time"]) & 
                             (data["trkfit"]["trksegs"]["time"] < 1650))
            
                # trk-level definition (the actual cut)
                within_t0 = ak.all(~trk_front | within_t0, axis=-1)
                cut_manager.add_cut( # trk-level
                    name="within_t0",
                    description="t0 at tracker entrance (640 < t_0 < 1650 ns)",
                    mask=within_t0
                )
                
            # 5. Track quality
            good_trkqual = selector.select_trkqual(data["trk"], quality=0.8)
            cut_manager.add_cut(
                name="good_trkqual",
                description="Track quality (quality > 0.8)",
                mask=good_trkqual
            )
            
            # 6. Loop helix maximum radius
            within_lhr_max = ((450 < data["trkfit"]["trksegpars_lh"]["maxr"]) & 
                              (data["trkfit"]["trksegpars_lh"]["maxr"] < 680)) # changed from 650
        
            # trk-level definition (the actual cut)
            within_lhr_max = ak.all(~trk_front | within_lhr_max, axis=-1)
            cut_manager.add_cut(
                name="within_lhr_max",
                description="Loop helix maximum radius (450 < R_max < 680 mm)",
                mask=within_lhr_max
            )
            
            # 7. Distance from origin
            within_d0 = (data["trkfit"]["trksegpars_lh"]["d0"] < 100)
        
            # trk-level definition (the actual cut)
            within_d0 = ak.all(~trk_front | within_d0, axis=-1) 
            cut_manager.add_cut(
                name="within_d0",
                description="Extrapolated position (d_0 < 100 mm)",
                mask=within_d0
            )
            
            # 8. Pitch angle
            within_pitch_angle = ((0.5577350 < data["trkfit"]["trksegpars_lh"]["tanDip"]) & 
                                  (data["trkfit"]["trksegpars_lh"]["tanDip"] < 1.0))
        
            # trk-level definition (the actual cut) 
            within_pitch_angle = ak.all(~trk_front | within_pitch_angle, axis=-1)
            cut_manager.add_cut(
                name="within_pitch_angle",
                description="Extrapolated pitch angle (0.5577350 < tan(theta_Dip) < 1.0)",
                mask=within_pitch_angle
            )

            # Mark CE-like tracks 
            # Useful for debugging 
            data["CE_like"] = cut_manager.combine_cuts()
            
            # cut_manager.add_cut(
            #     name="CE_like",
            #     description="CE-like tracks",
            #     mask=data["CE_like"],
            #     active=False
            # )
            
            # 9. CRV veto: |dt| < 150 ns (dt = coinc time - track t0) 
            dt_threshold = 150  
        
            # Get min and max track t0 times for each track
            min_trk_time = ak.min(data["trkfit"]["trksegs"]["time"][trk_front], axis=-1)
            max_trk_time = ak.max(data["trkfit"]["trksegs"]["time"][trk_front], axis=-1)
        
            # Get min and max coincidence times for each event
            min_coinc_time = ak.min(data["crv"]["crvcoincs.time"], axis=-1)
            max_coinc_time = ak.max(data["crv"]["crvcoincs.time"], axis=-1)
        
            # Broadcast coincidence times to match track times structure
            # we only care about the difference from t0, but this is more general
            
            # Create arrays with the right structure
            coinc_info = ak.zip({
                "min_time": min_coinc_time,
                "max_time": max_coinc_time
            })
            
            trk_info = ak.zip({
                "min_time": min_trk_time,
                "max_time": max_trk_time
            })
            
            # Use cartesian product to align each coinc with each track
            matched = ak.cartesian({"coinc": coinc_info, "track": trk_info})
            
            # Check if dt if within threshold
            veto = (
                (abs(matched["coinc", "min_time"] - matched["track", "max_time"]) < dt_threshold) |
                (abs(matched["track", "min_time"] - matched["coinc", "max_time"]) < dt_threshold)
            )

            # Mark unvetoed tracks
            # Useful for debugging 
            data["unvetoed"] = ~veto # Is the copy needed? 
            
            cut_manager.add_cut(
                name="unvetoed",
                description="No veto: |dt| >= 150 ns",
                mask=data["unvetoed"],
                active=False
            )

            # Mark CE-like unvetoed tracks 
            # Useful for debugging 
            data["unvetoed_CE_like"] = cut_manager.combine_cuts()

            self.logger.log("All cuts defined", "success")
            
        except Exception as e:
            self.logger.log(f"Error defining cuts: {e}", "error") 
            return None  
        
    def apply_cuts(self, data, cut_manager, active_only=True): # mask): 

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
            trk_mask = cut_manager.combine_cuts(active_only=active_only)

            # OPTIONAL: select track segments at tracker entrance
            # This makes it easier to study background events in the end
            data_cut["trkfit"] = data_cut["trkfit"][data_cut["at_trk_front"]]
            
            # Select tracks
            self.logger.log("Selecting tracks", "max")
            data_cut["trk"] = data_cut["trk"][trk_mask]
            data_cut["trkfit"] = data_cut["trkfit"][trk_mask]
            data_cut["trkmc"] = data_cut["trkmc"][trk_mask]
            
            # Then clean up events with no tracks after cuts
            self.logger.log(f"Cleaning up events with no tracks after cuts", "max") 
            evt_mask = ak.any(trk_mask, axis=-1)
            data_cut = data_cut[evt_mask] 
            
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

        hist_labels = ["All tracks", "CE-like tracks", "Unvetoed CE-like tracks"]

        try: 

            #### Create histogram objects:
            # Full momentum range histogram
            hist_full_range = hist.Hist(
                hist.axis.Regular(30, 0, 300, name="momentum", label="Momentum [MeV/c]"),
                hist.axis.StrCategory(hist_labels, name="selection", label="Selection")
            )
            # Signal region histogram (fine binning)
            hist_signal_region = hist.Hist(
                hist.axis.Regular(13, 103.6, 104.9, name="momentum", label="Momentum [MeV/c]"),
                hist.axis.StrCategory(hist_labels, name="selection", label="Selection")
            )

            # Process and fill histograms in batches
            def _fill_hist(trksegs, label): 
                """ Nested helper function to fill hists """
                mom = vector.get_mag(trksegs, "mom")
                # Flatten 
                if mom is None:
                    mom = ak.Array([])
                else:
                    mom = ak.flatten(mom, axis=None)
                    
                # Fill histogram for "all events"
                hist_full_range.fill(momentum=mom, selection=np.full(len(mom), label))

                # Filter for signal region
                mom_sig = mom[(mom >= 103.6) & (mom <= 104.9)]
                hist_signal_region.fill(momentum=mom_sig, selection=np.full(len(mom_sig), label))
            
                # Clean up 
                del mom, mom_sig 
                import gc
                gc.collect()
                    
        
            # 1. First process "all tracks" data
            _fill_hist(data["trkfit"]["trksegs"][data["at_trk_front"]], "All tracks") # This one needs the trk entrance cut
            _fill_hist(data_CE["trkfit"]["trksegs"], "CE-like tracks")
            _fill_hist(data_CE_unvetoed["trkfit"]["trksegs"], "Unvetoed CE-like tracks")
    
            self.logger.log("Histograms filled", "success")

            # Create a copy in results and explicitly delete large arrays after use
            result = {
                "Wide range": hist_full_range.copy(), 
                "Signal region": hist_signal_region.copy()
            }

            return result 

        except Exception as e:
            # Handle any errors that occur during processing
            self.logger.log(f"Error filling histograms: {e}", "error")
            return None
        
    def execute(self, data, file_id):
        """Perform complete analysis on an array
        
        Args:
            data: The data to analyse
            file_id: Identifier for the file
            
        Returns:
            dict: Complete analysis results
        """

        self.logger.log(f"Beginning analysis execution for file: {file_id}", "info") 
        
        try:
            # Create a unique cut manager for this file
            cut_manager = CutManager(verbosity=self.verbosity)
            
            # Define cuts
            self.logger.log("Defining cuts", "max")
            self.define_cuts(data, cut_manager)

            # Calculate cut stats
            self.logger.log("Getting cut stats", "max")
            cut_stats = cut_manager.calculate_cut_stats(data, progressive=True)

            # Apply CE-like cuts
            self.logger.log("Applying cuts", "max")
            # Not very efficient to do it this way...
            data_CE = self.apply_cuts(data, cut_manager, active_only=True) # Just CE-like tracks 
            data_CE_unvetoed = self.apply_cuts(data, cut_manager, active_only=False) # Unvetoed CE-like tracks
            
            # # Create histograms
            self.logger.log("Creating histograms", "max")
            histograms = self.create_histograms(data, data_CE, data_CE_unvetoed)
            
            # Compile all results
            self.logger.log("Analysis completed", "success")

            result = {
                "file_id": file_id,
                "cut_stats": cut_stats,
                "filtered_data": data_CE_unvetoed, # ,
                "histograms": histograms
            }

            # Force garbage collection
            gc.collect()
            
            return result
            
        except Exception as e:
            self.logger.log(f"Error during analysis execution: {e}", "error")  
            return None

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