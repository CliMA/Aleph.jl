# #= To run:
# nsys profile --trace=nvtx,cuda,mpi --output=report_%q{PMI_RANK} --force-overwrite true julia --project bin/test_cuda_mpi.jl 
# nsys stats --report cuda_gpu_trace report_0.nsys-rep --force-export true
# =#

using CUDA
using MPI
using Test

function test_gpu_visibility()
    println("Testing GPU Visibility")
    
    if CUDA.functional() && CUDA.has_cuda_gpu()
        x = CUDA.ones(1000)
        y = 2 .* x
        result = sum(y) |> Int
        
        @test result == 2000
        println("Basic GPU computation successful")
    else
        println("No CUDA device detected")
    end
end

function test_mpi(ArrayType = Array, comm = MPI.COMM_WORLD)
    rank = MPI.Comm_rank(comm)
    MPI.Comm_size(comm) < 2 && error("At least 2 MPI processes required")
    
    println("Testing MPI with ", ArrayType)

    MPI.Barrier(comm)
    N = 1024^2
    
    if rank == 0
        send_data = ArrayType{Float32}(undef, N); send_data .= 1.0f0
        recv_data = ArrayType{Float32}(undef, N); recv_data .= 0.0f0
        MPI.Send(send_data, 1, 0, comm)
        MPI.Recv!(recv_data, 1, 1, comm)
        @test all(recv_data .== 2.0f0)
        println("Transfer successful")
    elseif rank == 1
        send_data = ArrayType{Float32}(undef, N); send_data .= 2.0f0
        recv_data = ArrayType{Float32}(undef, N); recv_data .= 0.0f0
        MPI.Recv!(recv_data, 0, 0, comm)
        MPI.Send(send_data, 0, 1, comm)
        @test all(recv_data .== 1.0f0)
    else
        MPI.Barrier(comm)
    end
    CUDA.synchronize()

    MPI.Barrier(comm)
end

if !MPI.Initialized()
    println("Initializing MPI")
    try
        MPI.Init()
    catch e
        println("MPI Initialization Error")
        rethrow()
    end
end

rank = MPI.Comm_rank(MPI.COMM_WORLD)
println("MPI Rank: $rank")
if rank == 0    
    MPI.versioninfo()
    CUDA.versioninfo()
end

test_gpu_visibility()
test_mpi(Array)
test_mpi(CuArray)

println("All tests completed!")
MPI.Finalize()

