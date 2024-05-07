using BenchmarkTools

x_start = 4
x_end   = 300
n_data  = 25

x_flt = 10 .^(range(log10(x_start), stop = log10(x_end), length = n_data))      # x_flt = float values of normal x-values
x_rnd = round.(Int, x_flt)                                                      # x_rnd = rounded x-values in normal scale
x_log = log10.(x_rnd)                                                           # x_log = x-values in log scale

# Code to do benchmarks of a nxn matrix for each n in the interval

i = 1
for n in x_rnd

    println("Matrix size: ", n, " running")

    A = rand(n,n);
    Cache_LU = lu!(A)

    folder_path = "/Users/evakuipers/Jems.jl/BM_outputfolders/BM_LU_output"      # lol change this
    bm = @benchmark $Cache_LU = lu!($A)

    file_path = joinpath(folder_path, "BM_LU_n$(n).json")
    BenchmarkTools.save(file_path, bm)

    println("Benchmark ", i, " done")

    i += 1

end