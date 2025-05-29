import uproot
import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import hist
import gc

from pyselect import Select
from pyvector import Vector
from cut_manager import CutManager

class Analyse:
    """Class to handle analysis functions
    """
    def __init__(self, verbosity=1):
        """Initialise the analysis handler
        
        Args:
            verbosity (int, optional): Level of output detail (0: critical errors only, 1: info, 2: debug, 3: deep debug)
        """
        self.verbosity = verbosity
        self.print_prefix = "[Analyse] "
        
        # Initialise tools
        self._log(f"Initialising selector and vector tools", level=1)
        self.selector = Select(verbosity=self.verbosity)
        self.vector = Vector(verbosity=self.verbosity) 
        
        # Analysis configuration
        self.on_spill = False  # Default to off-spill 
        self._log(f"Initialised with on_spill={self.on_spill}", level=1)
    
    def _log(self, message, level=1): # now depracated! 
        """Print a message based on verbosity level
        
        Args:
            message (str): The message to print
            level (int): Minimum verbosity level required to print this message
        """
        if self.verbosity >= level:
            print(f"{self.print_prefix}{message}")
    
    def define_cuts(self, data, cut_manager, on_spill=None):
        """Define analysis cuts
        
        Args:
            data (ak.Array): data to apply cuts to
            cut_manager: The CutManager instance to use
            on_spill (bool, optional): Whether to apply on-spill specific cuts
        """
        self._log(f"Defining cuts (on_spill={on_spill if on_spill is not None else self.on_spill})", level=1)
        
        if on_spill is None:
            on_spill = self.on_spill
            
        selector = self.selector
        
        try:
            # Tracker surfaces
            trk_front = selector.select_surface(data["trkfit"], sid=0)
            trk_mid = selector.select_surface(data["trkfit"], sid=1)
            trk_back = selector.select_surface(data["trkfit"], sid=2)
            in_trk = (trk_front | trk_mid | trk_back)
            
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
            self._log("Defining downstream tracks cut", level=2)
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
                self._log("Defining time cut (on-spill specific)", level=2)
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
            
            cut_manager.add_cut(
                name="CE_like",
                description="CE-like tracks",
                mask=data["CE_like"],
                active=False
            )
            
            # 9. CRV veto: |dt| < 150 ns (dt = coinc time - track t0) 
            dt_threshold = 150  
        
            # Get min and max track t0 times for each track
            min_trk_time = ak.min(data["trkfit"]["trksegs"]["time"][trk_front], axis=-1)
            max_trk_time = ak.max(data["trkfit"]["trksegs"]["time"][trk_front], axis=-1)
        
            # Get min and max coincidence times for each event
            min_coinc_time = ak.min(data["crv"]["crvcoincs.time"], axis=-1)
            max_coinc_time = ak.max(data["crv"]["crvcoincs.time"], axis=-1)
        
            # Broadcast coincidence times to match track times structure
            
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
                (abs(matched["coinc", "min_time"] - matched["track", "max_time"]) < dt_threshold) & 
                (abs(matched["track", "min_time"] - matched["coinc", "max_time"]) < dt_threshold)
            )

            # Mark unvetoed tracks
            # Useful for debugging 
            data["unvetoed"] = ~veto # Is the copy needed? 
            
            cut_manager.add_cut(
                name="unvetoed",
                description="No veto: |dt| >= 150 ns",
                mask=data["unvetoed"]
            )

            self._log("All cuts defined successfully", level=1)
            
        except Exception as e:
            self._log(f"Error defining cuts: {e}", level=0)  # Always print errors
            raise  # Re-raise the exception
        
    def apply_cuts(self, data, cut_manager): # mask): 
        # FIXME! 
        """Apply all trk-level mask to the data
        
        Args:
            data: Data to apply cuts to
            mask: Mask to apply 
            
        Returns:
            ak.Array: Data after cuts applied
        """
        self._log("Applying cuts to data", level=1)
        
        try:
            # Make an empty container for filtered data
            data_cut = {}             
            
            # Combine cuts
            self._log("Combining cuts", level=2)
            # CE_like = data["CE_like"] # .combine_cuts(active_only=True) # unvetoed is inactive
            # unvetoed = cut_manager.combine_cuts(active_only=False) # unvetoed in inactive
            trk_mask = cut_manager.combine_cuts(active_only=False)
            # data_cut["combined"] = combined
            
            # # Select tracks
            self._log("Selecting tracks", level=2)
            data_cut["trk"] = data["trk"][trk_mask]
            data_cut["trkfit"] = data["trkfit"][trk_mask]
            data_cut["trkmc"] = data["trkmc"][trk_mask]
            
            # Then clean up events with no tracks after cuts
            self._log("Cleaning up events with no tracks after cuts", level=2)
            evt_mask = ak.any(trk_mask, axis=-1)
            data_cut = data[evt_mask] 
            
            self._log("Cuts applied successfully", level=1)
            
            return data_cut
            
        except Exception as e:
            self._log(f"Error applying cuts: {e}", level=0)  # Always print errors
            raise

    def create_histograms(self, data, data_cut):
        """Create histograms from the data before and after applying cuts
        
        Args:
            data: Data before cuts
            data_cut: Data after cuts
            
        Returns:
            dict: Dictionary of histograms
        """
        self._log("Creating histograms", level=1)
        
        # Tools 
        selector = self.selector 
        vector = self.vector

        try: 

            #### Create histogram objects:
            # Full momentum range histogram
            hist_full_range = hist.Hist(
                hist.axis.Regular(1000, 0, 1000, name="momentum", label="Momentum [MeV/c]"),
                hist.axis.StrCategory(["All tracks", "CE-like tracks"], name="selection", label="Selection")
            )
            # Signal region histogram (fine binning)
            hist_signal_region = hist.Hist(
                hist.axis.Regular(13, 103.6, 104.9, name="momentum", label="Momentum [MeV/c]"),
                hist.axis.StrCategory(["All tracks", "CE-like tracks"], name="selection", label="Selection")
            )

            # Process and fill histograms in batches
        
            # 1. First process "before cuts" data
            self._log("Processing 'wide range' events", level=2)
            at_trkent_all = selector.select_surface(data["trkfit"], sid=0)
            mom_all = vector.get_mag(data["trkfit"]["trksegs"][at_trkent_all], "mom")
            
            # Flatten 
            if mom_all is None:
                mom_all = ak.Array([])
            else:
                mom_all = ak.flatten(mom_all, axis=None)
                
            # Fill histogram for "all events"
            hist_full_range.fill(momentum=mom_all, selection=np.full(len(mom_all), "All tracks"))
            
            # Filter for signal region
            mom_all_sig = mom_all[(mom_all >= 103.6) & (mom_all <= 104.9)]
            hist_signal_region.fill(momentum=mom_all_sig, selection=np.full(len(mom_all_sig), "All tracks"))
            
            # Clean up to free memory
            del mom_all, mom_all_sig, at_trkent_all
            import gc
            gc.collect()

            # 2. Now process "after cuts" data 
            self._log("Processing 'signal region' events", level=2)
            at_trkent_cut = selector.select_surface(data_cut["trkfit"], sid=0)
            mom_cut = vector.get_mag(data_cut["trkfit"]["trksegs"][at_trkent_cut], "mom")
            
            # Convert to empty array if needed
            if mom_cut is None:
                mom_cut = ak.Array([])
            else:
                mom_cut = ak.flatten(mom_cut, axis=None)
                
            # Fill histogram for "CE-like events"
            hist_full_range.fill(momentum=mom_cut, selection=np.full(len(mom_cut), "CE-like tracks"))
            
            # Filter for signal region
            mom_cut_sig = mom_cut[(mom_cut >= 103.6) & (mom_cut <= 104.9)]
            hist_signal_region.fill(momentum=mom_cut_sig, selection=np.full(len(mom_cut_sig), "CE-like tracks"))
            
            # Clean up to free memory
            del mom_cut, mom_cut_sig, at_trkent_cut
            gc.collect()
        
            # # Data
            # self._log("Selecting tracker entrance surfaces", level=2)
            # at_trkent_all = selector.select_surface(data["trkfit"], sid=0)
            # at_trkent_cut = selector.select_surface(data_cut["trkfit"], sid=0)
            
            # self._log("Calculating momentum magnitudes", level=2)
            # mom_all = vector.get_mag(data["trkfit"]["trksegs"][at_trkent_all], "mom")
            # mom_cut = vector.get_mag(data_cut["trkfit"]["trksegs"][at_trkent_cut], "mom")

            # # Flatten, but check None and convert to empty array if needed
            # mom_all = ak.flatten(mom_all, axis=None) if mom_all is not None else ak.Array([])
            # mom_cut = ak.flatten(mom_cut, axis=None) if mom_cut is not None else ak.Array([])

            # # Define histograms 
            # self._log("Creating histogram objects", level=2)
    

            # ##### Fill the histograms #####
            # self._log("Filling histograms", level=2)
    
            # # Full range histogram
            # hist_full_range.fill(momentum=mom_all, selection=np.full(len(mom_all), "All events"))
            # hist_full_range.fill(momentum=mom_cut, selection=np.full(len(mom_cut), "CE-like events"))
            
            # # Signal region histogram
            # # For signal region, we need to filter the momenta to be within the range
            # mom_all_sig = mom_all[(mom_all >= 103.6) & (mom_all <= 104.9)]
            # mom_cut_sig = mom_cut[(mom_cut >= 103.6) & (mom_cut <= 104.9)]
            
            # if self.verbosity >= 2:
            #     self._log(f"Signal region events: all={len(mom_all_sig)}, cut={len(mom_cut_sig)}", level=2)
            
            # hist_signal_region.fill(momentum=mom_all_sig, selection=np.full(len(mom_all_sig), "All events"))
            # hist_signal_region.fill(momentum=mom_cut_sig, selection=np.full(len(mom_cut_sig), "CE-like events"))
    
            self._log("Histograms filled successfully", level=1)

            # Create a copy in results and explicitly delete large arrays after use
            result = {
                "Wide range": hist_full_range.copy(), 
                "Signal region": hist_signal_region.copy()
            }
        
            # Force cleanup of large references
            # del mom_all, mom_cut, mom_all_sig, mom_cut_sig

            return result 

        except Exception as e:
            # Handle any errors that occur during processing
            self._log(f"Error filling histograms: {e}", level=0)  # Always print errors
            return None
        
    def execute(self, data, file_id):
        """Perform complete analysis on an array
        
        Args:
            data: The data to analyse
            file_id: Identifier for the file
            
        Returns:
            dict: Complete analysis results
        """
        self._log(f"Beginning analysis execution for file: {file_id}", level=1)
        
        try:
            # Create a unique cut manager for this file
            cut_manager = CutManager(verbosity=self.verbosity)
            
            # Define cuts
            self._log("Defining cuts", level=1)
            self.define_cuts(data, cut_manager)

            # Calculate cut stats
            self._log("Getting cut stats", level=1)
            cut_stats = cut_manager.calculate_cut_stats(data, progressive=True)

            # Retrieve masks
            CE_like = data["CE_like"] # copy 

            # Apply CE-like cuts
            self._log("Applying cuts", level=1)
            data_cut = self.apply_cuts(data, cut_manager)
            
            # Create histograms
            self._log("Creating histograms", level=1)
            histograms = self.create_histograms(data, data_cut)
            
            # Compile all results
            self._log("Analysis completed successfully", level=1)

            result = {
                "file_id": file_id,
                "cut_stats": cut_stats,
                "filtered_data": data_cut,
                "histograms": histograms,
            }

            # Force garbage collection
            gc.collect()
            
            return result
            
        except Exception as e:
            self._log(f"Error during analysis execution: {e}", level=0)  # Always print errors
            raise


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
            raise