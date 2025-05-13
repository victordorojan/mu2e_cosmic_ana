# from cut_manager import CutManager

import json
import awkward as ak
import csv

# Can move this into pyselect
class CutManager:
    """Class to manage analysis cuts"""
    
    def __init__(self, verbosity=1):
        """Initialise 
        
        Args:
            verbosity (int, optional): Printout level (0: minimal, 1: normal, 2: detailed)
        """

        # Initilaise parameters
        self.verbosity = verbosity
        self.print_prefix = "[pyselect] "
        
        # Initialise cut container
        self.cuts = {}

    def _log(self, message, level=1):
        """Print a message based on verbosity level
        
        Args:
            message (str): The message to print
            level (int): Minimum verbosity level required to print this message
        """
        if self.verbosity >= level:
            print(f"{self.print_prefix}{message}")
    
    def add_cut(self, name, description, mask, active=True):
        """
        Add a cut to the collection.
        
        Args: 
            name (str): Name of the cut
            description (str): Description of what the cut does
            mask (awkward.Array): Boolean mask array for the cut
            active (bool, optional): Whether the cut is active by default
        """

        # Get the next available index
        next_idx = len(self.cuts)
    
        self.cuts[name] = {
            "description": description,
            "mask": mask,
            "active": active,
            "idx" : next_idx
        }

        self._log(f"{self.print_prefix}Added cut {name} with index {next_idx}", level=2)
        # return self
        # This would allow method chaining, could be useful maybe?
    
    def get_cut_mask(self, name):
        """Utility to return the Boolean mask for a specific cut.
        
        Args:
            name (str): Name of the cut
            
        """
        if name in self.cuts:
            return self.cuts[name]["mask"]
        else:
            raise ValueError(f"{self.print_prefix}Cut '{name}' not defined")
    
    def set_active_cut(self, name, active=True):
        """Utility to set a cut as active or inactive 

        Args: 
            name (str): Name of the cut
            active (bool, optional): Whether the cut should be active. Default is True.
        """
        if name in self.cuts:
            self.cuts[name]["active"] = active
        else:
            raise ValueError(f"{self.print_prefix}Cut '{name}' not defined")
            
    def get_active_cuts(self):
        """Utility to get all active cutss"""
        return {name: cut for name, cut in self.cuts.items() if cut["active"]}
    
    def combine_cuts(self, cut_names=None, active_only=True):
        """ Return a Boolean combined mask from specified cuts. Applies an AND operation across all cuts. 
        Args: 

        cut_names (list, optional): List of cut names to include (if None, use all cuts)
        active_only (bool, optional): Whether to only include active cuts
        """
        if cut_names is None:
            # Then use all cuts in original order
            cut_names = list(self.cuts.keys())
        # Init mask
        combined = None
        # Loop through cuts        
        for name in cut_names:
            # Get info dict for this cut
            cut_info = self.cuts[name]
            # Active cuts
            if active_only and not cut_info["active"]:
                continue
            # If first cut, initialise 
            if combined is None:
                combined = cut_info["mask"]
            else:
                combined = combined & cut_info["mask"] 
        
        return combined
    
    def calculate_cut_stats(self, data, progressive=True, active_only=False):
        """ Utility to calculate stats for each cut.
        
        Args:
            data (awkward.Array): Input data 
            progressive (bool, optional): If True, apply cuts progressively; if False, apply each cut independently. Default is True.
        """
        total_events = len(data)
        stats = []
        
        # Base statistics (no cuts)
        stats.append({
            "name": "No cuts",
            "active": "N/A",
            "description": "No selection applied",
            "events_passing": total_events,
            "absolute_frac": 100.0,
            "relative_frac": 100.0
        })
        
        # Get cuts 
        cuts = [name for name in self.cuts.keys()] 
        if active_only: 
            cuts = [name for name in self.cuts.keys() if self.cuts[name]["active"]]
        
        previous_mask = None
        
        for name in cuts:
            cut_info = self.cuts[name]
            mask = cut_info["mask"]
            
            if progressive and previous_mask is not None:
                # Apply this cut on top of previous cuts
                current_mask = previous_mask & mask
            else:
                # Apply this cut independently
                current_mask = mask
            
            # Calculate event-level efficiency
            event_mask = ak.any(current_mask, axis=-1) # events that have ANY True combined mask
            events_passing = ak.sum(event_mask) # Count up these events
            absolute_frac = events_passing / total_events * 100
            relative_frac = (events_passing / stats[-1]["events_passing"] * 100 
                           if stats[-1]["events_passing"] > 0 else 0)
            
            stats.append({
                "name": name,
                "active": cut_info["active"],
                "description": cut_info["description"],
                "mask": current_mask,
                "events_passing": int(events_passing),
                "absolute_frac": float(absolute_frac),
                "relative_frac": float(relative_frac)
            })
            
            if progressive:
                # Update for next iteration
                previous_mask = current_mask
        
        return stats
    
    def print_cut_stats(self, data=None, stats=None, progressive=True, active_only=False, csv_name=None):
        """ Print cut statistics for each cut.
        
        Args:
            data (awkward.Array): Data array
            progressive (bool, optional): If True, apply cuts progressively; if False, apply each cut independently
        """

        # Input validation
        sources = sum(x is not None for x in [data, stats]) 
        if sources != 1:
            self._log(f"Please provide exactly one of 'data' or 'stats'", level=0)
            return None

        if not stats:
            stats = self.calculate_cut_stats(data, progressive, active_only)


        # Print header
        print(f"\n{self.print_prefix}Cut Info:")
        print("-" * 110)
        header = "{:<20} {:<10} {:<20} {:<20} {:<20} {:<30}".format(
            "Cut", "Active", "Events Passing", "Absolute Frac. [%]", "Relative Frac. [%]", "Description")
        print(header)
        print("-" * 110)
        # Print each cut's statistics
        for stat in stats:
            row = "{:<20} {:<10} {:<20} {:<20.2f} {:<20.2f} {:<30}".format(
                stat["name"], 
                stat["active"],
                stat["events_passing"], 
                stat["absolute_frac"], 
                stat["relative_frac"], 
                stat["description"])
            print(row)
        print("-" * 110)
        
        # Print final statistics
        if len(stats) > 1:
            first_events = stats[0]["events_passing"]
            last_events = stats[-1]["events_passing"]
            overall_eff = last_events / first_events * 100 if first_events > 0 else 0
            
            self._log(f"Summary: {last_events}/{first_events} events remaining ({overall_eff:.2f}%)", level=0)

        # Write to CSV if requested
        if csv_name:
            try:
                with open(csv_name, 'w', newline='') as csv_file:
                    # Create a CSV writer
                    writer = csv.writer(csv_file)
                    
                    # Write header
                    writer.writerow(['Cut', 'Active', 'Events Passing', 'Absolute Frac. [%]', 'Relative Frac. [%]', 'Description'])
                    
                    # Write data rows
                    for stat in stats:
                        writer.writerow([
                            stat["name"],
                            stat["active"],
                            stat["events_passing"],
                            f"{stat['absolute_frac']:.2f}",
                            f"{stat['relative_frac']:.2f}",
                            stat["description"]
                        ])
                    
                    self._log(f"Cut statistics written to {csv_name}", level=1)

            except Exception as e:
                self._log(f"Error writing cut statistics to CSV: {e}", level=0)
                
    def save_cuts(self, file_name):
        """ Save the current cut configuration to a JSON file.
        
        Args:
            file_name (str): File path to save the configuration
        """
        config = {
            "cuts": {}
        }
        
        for name, info in self.cuts.items():
            config["cuts"][name] = {
                "description": info["description"],
                "active": info["active"],
                "idx": info["idx"]
            }
        
        with open(file_name, 'w') as f:
            json.dump(config, f, indent=2)
            
        self._log(f"Saved cut configuration to {file_name}", level=1)

    def combine_cut_stats(self, stats_list):
        """Combine a list of cut statistics after multiprocessing 
        
        Args:
            cut_stats_list: List of cut statistics lists from different files
        
        Returns:
            list: Combined cut statistics
        """
        # Return empty list if no input
        if not stats_list:
            return []
        
        # Use the first list as a template
        combined_stats = []
        for cut in stats_list[0]:
            # Create a copy without the mask (which we don't need)
            cut_copy = {k: v for k, v in cut.items() if k != 'mask'}
            # Reset the event count
            cut_copy['events_passing'] = 0
            combined_stats.append(cut_copy)
        
        # Sum up events_passing for each cut across all files
        for stats in stats_list:
            for i, cut in enumerate(stats):
                if i < len(combined_stats):  # Safety check
                    combined_stats[i]['events_passing'] += cut['events_passing']
        
        # Recalculate percentages
        if combined_stats and combined_stats[0]['events_passing'] > 0:
            total_events = combined_stats[0]['events_passing']
            previous_events = total_events
            
            for i, cut in enumerate(combined_stats):
                events = cut['events_passing']
                
                # Absolute percentage
                cut['absolute_frac'] = (events / total_events) * 100.0
                
                # Relative percentage
                if i == 0:  # "No cuts"
                    cut['relative_frac'] = 100.0
                else:
                    prev_events = combined_stats[i-1]['events_passing']
                    cut['relative_frac'] = (events / prev_events) * 100.0 if prev_events > 0 else 0.0
        
        return combined_stats
