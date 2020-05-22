Here is how to perform kernel microbenchmarking in hb-pytorch:

(This only works in cosim, because emulation does not have print stat capabilities)

0. git checkout kz73_dummy_microbenchmark

1. Write kernel code in hb-pytorch/hammerblade/torch/kernel/kernel_dummy.cpp.

2. Write host code in hb-pytorch/aten/src/ATen/native/hammerblade/Dummy.cpp.

3. Edit configuration file at hb-pytorch/aten/src/ATen/native/native_functions.yaml. 
   Refer to vvadd as an example.

4. Add tests to hb-pytorch/hammerblade/torch/tests/test_dummy.cpp.

5. Edit hb-pytorch/hammerblade/torch/pytest_runner.py to run your newly added tests.

6. Remake pytorch by these commands at the top level:
   python setup.py clean
   python setup.py develop

7. Go to hb-pytorch/hammerblade/torch and run
   make clean
   make regression

   This will generate the vanilla_stats.csv file which we need to gather data from.

8. While still in hb-pytorch/hammerblad/torch, run
   python vanilla_stats_parser_with_graph.py --print_stats

   I've edited the original vanilla_stats_parser.py to add graphing capabilities and 
   easily printing stats to the terminal. (Note that this version of the script is only
   in my branch).
   This will print the execution cycle counts and instruction counts to the terminal.

9. Analyze the collected data! I've been using excel manually but feel free to automate this.

