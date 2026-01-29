from plotting.make_maze_plots import plot_occupancy, plot_propcorrect, plot_startplatforms
import os
directory_path = r""
most_recent_file = None
most_recent_time = 0

# iterate over the files in the directory using os.scandir
for entry in os.scandir(directory_path):
    if entry.is_file():
        # get the modification time of the file using entry.stat().st_mtime_ns
        mod_time = entry.stat().st_mtime_ns
        if mod_time > most_recent_time:
            # update the most recent file and its modification time
            most_recent_file = entry.name
            most_recent_time = mod_time
            
with open(most_recent_file, 'r') as f:  
   # read file contents into a variable 
   contents = f.read()  

   # do something with contents of file 

   # write contents back to file 
   f.write(contents)  

   # close file when done reading/writing  
   f.close()
   