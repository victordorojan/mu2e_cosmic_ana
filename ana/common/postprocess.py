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
        arrays = []
        
        # Check if we have results
        if not results:
            return None
            
        # Loop through all files
        for result in results: #
            array = ak.Array(result["filtered_data"])
            if len(array) == 0:
                continue
            # Concatenate arrays
            arrays.append(array)
            
        if len(arrays) == 0:
            self.logger.log(f"Combined array has zero length", "warning") 
            return arrays

        combined_array = ak.concatenate(arrays)
        
        self.logger.log(f"Combined arrays, result contains {len(combined_array)} events", "success")

        return combined_array

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

    def get_background_events(self, results): # , printout=True, out_path=None): 
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

        self.logger.log(f"Retrieved background event info", "success")

        return output 
        
        # # Print 
        # if printout:
        #     self.logger.log(f"Info for {count} background events :", "info")
        #     print(output)
        
        # # Write to file
        # if out_path:
        #     with open(out_path, "w") as f:
        #         f.write(output)
        
        #     self.logger.log(f"Wrote {out_path}", "success")


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
        combined_background_info = self.get_background_events(results)
        
        self.logger.log(f"Postprocessing complete:\n\treturning tuple of combined arrays, combined histograms, and combined cut stats", "success")
        return combined_array, combined_hists, combined_stats, combined_background_info