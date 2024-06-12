# Locus

You can run Locus using command line: `java main.Main [config]`
It reads all the configurations from the [config] file.

You can specify the config file to any file, which should have the same format as `config_example.txt`

The reauired configurations are as followings:

1. `task`: the task you want to run 

    * all: conduct all the previous step

2. `repoDir`: which refers to the data dir of your target project. 
3. `sourceDir`: which is the target version of the source files
4. `workingLoc`: the working directory of the current run, which will store all the intermediate files and the final results
5. `bugReport`: specify the bug report file
6. `changeOracle`: deprecated
6. `revisionsLoc`: which refers to the code commit location
