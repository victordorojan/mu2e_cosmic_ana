import sys
import gc
from datetime import datetime
import pickle

from pyutils.pyprocess import Processor, Skeleton
from pyutils.pylogger import Logger
logger = Logger(print_prefix="[run_cut_scan]")

sys.path.append("../common")
from analyse import Analyse
from postprocess import PostProcess

# CUT CONFIGS
from cut_configs import cut_configs

# Get today"s date in MM-DD-YY format
today = datetime.now().strftime("%m-%d-%y")
tag = "test_" + today
ds_type = "offspill"

# Create your custom processor class
class CosmicProcessor(Skeleton):
    """Your custom file processor 
    
    This class inherits from the Skeleton base class, which provides the 
    basic structure and methods withing the Processor framework 
    """
    def __init__(self, cuts_to_toggle):
        """Initialise your processor with specific configuration
        
        This method sets up all the parameters needed for this specific analysis.

        Args: 
            cuts_to_toggle (dict): Dict of cut name with active state         
        """
        # Call the parent class"s __init__ method first
        # This ensures we have all the base functionality properly set up
        super().__init__()

        # Now override parameters from the Skeleton with the ones we need
        # Data selection configuration 
        # self.defname = "nts.mu2e.CosmicCRYSignalAllOffSpillTriggered-LH.MDC2020as_best_v1_3_v06_03_00.root"
        self.file_name = "/exp/mu2e/data/users/sgrant/mu2e_cosmic_ana/data/nts.mu2e.CosmicCRYSignalAllOffSpillTriggered-LH.MDC2020as_best_v1_3_v06_03_00.001202_00050440.root"
        
        self.branches = { 
            "evt" : [
                "run",
                "subrun",
                "event",
            ],
            "crv" : [
                "crvcoincs.time",
                "crvcoincs.nHits",
                "crvcoincs.pos.fCoordinates.fZ"
            ],
            "trk" : [
                "trk.nactive", 
                "trk.pdg", 
                "trkqual.valid",
                "trkqual.result"
            ],
            "trkfit" : [
                "trksegs",
                "trksegpars_lh"
            ],
            "trkmc" : [
                "trkmcsim"
            ]
        }
        self.use_remote = False     # Use remote file via mdh
        self.location = "disk"     # File location
        self.max_workers = 1      # Limit the number of workers
        self.verbosity = 2         # Set verbosity 
        self.use_processes = True  # Use processes rather than threads
        
        # Now add your own analysis-specific parameters 

        # Init analysis methods
        # Would be good to load an analysis config here 
        self.analyse = Analyse(
            # event_subrun=(93561, 25833), # select one event
            on_spill=False,
            verbosity=0
        )

        self.logger = Logger(
            print_prefix = "[CosmicProcessor]"
        )
            
        # Toggle cuts 
        self.cuts_to_toggle = cuts_to_toggle
            
        # Custom prefix for log messages from this processor
        self.logger.log(f"Initialised with {cuts_to_toggle}", "success")
        
    # ==========================================
    # Define the core processing logic
    # ==========================================
    # This method overrides the parent class"s process_file method
    # It will be called automatically for each file by the execute method
    def process_file(self, file_name): 
        """Process a single ROOT file
        
        This method will be called for each file in our list.
        It extracts data, processes it, and returns a result.
        
        Args:
            file_name: Path to the ROOT file to process
            
        Returns:
            A tuple containing the histogram (counts and bin edges)
        """
        try:
            # Create a local pyprocess Processor to extract data from this file
            # This uses the configuration parameters from our class
            this_processor = Processor(
                use_remote=self.use_remote,     # Use remote file via mdh
                location=self.location,         # File location
                verbosity=0 # self.verbosity        # Reduce output in worker threads
            )
            
            # Extract the data 
            this_data = this_processor.process_data(
                file_name = file_name, 
                branches = self.branches
            )
            
            # ---- Analysis ----            
            results = self.analyse.execute(
                this_data, 
                file_name,
                self.cuts_to_toggle
            )

            # Clean up
            gc.collect()

            return results 
        
        except Exception as e:
            # Handle any errors that occur during processing
            self.logger.log(f"Error processing {file_name}: {e}", "success")
            return None

def save_results(results, out_path):
    """Save results dictionary to pickle file"""
    
    logger.log(f"Saving results to {out_path}", "info")
    
    try:
        with open(out_path, "wb") as f:
            pickle.dump(results, f)
        
        logger.log(f"Pickle successful", "success")
        
    except Exception as e:
        logger.log(f"Pickle failed: {e}", "error")
        
def main(): 
        
    for i_cut, cut_config in enumerate(cut_configs): 
    
        # Printout 
        logger.log(f"Running cut config: {cut_config}", "info")

        # Handle nomimal case
        cuts_to_toggle = None if "nominal" in cut_config else cut_config
    
        # Initalise Processor 
        cosmic_processor = CosmicProcessor(cuts_to_toggle = cuts_to_toggle)
    
        # Execute analysis
        analysis_results = cosmic_processor.execute()
    
        # Initialise postprocessor 
        postprocessor = PostProcess()
    
        # Execute postprocessor
        data, hists, stats, info = postprocessor.execute(analysis_results)
    
        # Return dict 
        results_to_save = {
            "config_id": i_cut,
            "config": cut_config,
            "data": data,
            "hists": hists,
            "stats": stats,
            "info": info
        }

        save_results(
            results_to_save,
            f"../../pkl/{ds_type}/results_cut_config_{i_cut}_{tag}.pkl"
        )

        logger.log(f"Completed config {i_cut+1}/{len(cut_configs)}", "success")

    logger.log(f"Done", "success") 

if __name__ == "__main__":
    main()