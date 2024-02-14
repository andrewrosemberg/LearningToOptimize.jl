#= Julia code for launching jobs on the slurm cluster.

This code is expected to be run from an sbatch script after a module load julia command has been run.
It starts the remote processes with srun within an allocation.
If you get an error make sure to Pkg.checkout("CluterManagers").
=#

try

	using Distributed, ClusterManagers
catch
	Pkg.add("ClusterManagers")
	Pkg.checkout("ClusterManagers")
end

using Distributed, ClusterManagers

np = 50 #
addprocs(SlurmManager(np), job_file_loc = ARGS[1]) #cpus_per_task=24, mem_per_cpu=24, partition="debug", t="08:00:00")

println("We are all connected and ready.")

include(ARGS[2])
