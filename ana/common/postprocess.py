import awkward as ak
from pyutils.pylogger import Logger
from cut_manager import CutManager

class PostProcess():
    """Class for postprocessing
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
        self.logger.log(f"Initialised", "info")

    def combine_arrays(self, results):
        """Combine filtered arrays from multiple files
        """
        combined_array = []
        
        # Check if we have results
        if not results:
            return None
            
        # Loop through all files
        for result in results: #
            array = ak.Array(result["filtered_data"])
            if len(array) == 0:
                continue
            # Concatenate arrays
            combined_array.append(array)
            
        if len(combined_array) == 0:
            self.logger.log(f"Combined array has zero length", "warning") 
        else:
            self.logger.log(f"Combined arrays, result contains {len(combined_array)} events", "success")
            
        return ak.concatenate(combined_array)

    def combine_hists(self, results):
        """Combine histograms
        
        Args:
            results: List of results per file
            
        Returns:
            dict: Combined histograms
        """
        combined_hists = {}
        
        # Check if we have results
        if not results:
            return None
        
        # Loop through all files
        for result in results: # 
            # Skip if no histograms in this file
            if 'histograms' not in result or not result['histograms']:
                continue
            
            # Process each histogram type
            for hist_name, hist_obj in result['histograms'].items():
                if hist_name not in combined_hists:
                    # First time seeing this histogram type, initialise
                    combined_hists[hist_name] = hist_obj.copy()
                else:
                    # Add this histogram to the accumulated one
                    combined_hists[hist_name] += hist_obj
                    
        self.logger.log(f"Combined {len(combined_hists)} histograms over {len(results)} results", "success")
        return combined_hists

    def combine_cut_stats(self, results):
        """
        Combine cuts stats into a list, then combine the cuts with CutManager
        """
        stats = [] 
        if isinstance(results, list): 
            for result in results: 
                if "cut_stats" in result: 
                    stats.append(result["cut_stats"])
        else: 
            stats.append(results["cut_stats"])

        cut_manager = CutManager()
        combined_stats = cut_manager.combine_cut_stats(stats)

        self.logger.log(f"Combined cut statistics", "success")
        return combined_stats

    def execute(self, results): 
        """ 
        Args: 
            results (list): list of results
        Returns:
            tuple of combined arrays and combined histograms
        """
        # This handles single files 
        if not isinstance(results, list):
            results = [results]
    
        combined_array = self.combine_arrays(results)
        combined_hists = self.combine_hists(results)
        combined_stats = self.combine_cut_stats(results)
        
        self.logger.log(f"Postprocessing complete:\n\treturning tuple of combined arrays, combined histograms, and combined cut stats", "success")
        return combined_array, combined_hists, combined_stats