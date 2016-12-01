using HiddenMarkovChains
using Distributions
using Base.Test

include("utilities.jl")

######################################################################
# helper functions
######################################################################

@testset "Validating probability vectors and matrices." begin

    @test HiddenMarkovChains.isprobvec([0.1, 0.2, 0.7])
    @test !HiddenMarkovChains.isprobvec([-0.1, 0.1, 1])
    @test !HiddenMarkovChains.isprobvec([0.1, 0.2])

    @test HiddenMarkovChains.isprobmat(ones(2,3)/3)
    @test !HiddenMarkovChains.isprobmat([1.0 2.0;])
    @test !HiddenMarkovChains.isprobmat([-1.0 2.0])

end

@testset "Iteration over nonzeros." begin

    @test collect(HiddenMarkovChains.column_nz(reshape(1:9, 3, 3), 2)) ==
        [(1,4), (2,5), (3,6)]
    @test collect(HiddenMarkovChains.column_nz(sparse([1,1,3,2],
                                                      [1,2,2,3],
                                                      [2,3,5,7]), 2)) ==
                                                          [(1,3), (3,5)]
end

######################################################################
# rowsums
######################################################################

@testset "Rowsums." begin
    for i in 1:100
        m = randn(10,10)
        enforce_rowsums!(m)
        @test sum(m, 2) ≈ ones(Float64, size(m,1), 1)
    end
end

######################################################################
# TEST: steady state calculations
######################################################################

@testset "Steady state calculations." begin

    @test_throws DimensionMismatch steady_state(ones(2,3)/3)

    "Steady state calculation error."
    function ss_error(P)
        p = steady_state(P)
        norm(p'*P-p', 1)
    end

    for i in 1:1000
    @test ss_error(random_sparse_stochastic_matrix(9, 4)) < 1e-10
    end

    for i in 1:1000
        @test ss_error(random_stochastic_matrix(9)) < 1e-10
    end

end

@testset "Path simulations." begin

    let h = random_HMC(3, 5, 3),
        pp = path_probabilities(h),
        pp_sim = simulate_path_probabilities(h, 10000),
        diffs = path_probabilities_difference(pp, pp_sim)
        @test diffs < (0.01,0.01)
        @test length(uncovered_paths(pp, pp_sim)) == 0
    end

    let h = random_HMC(3, 5, 3),
        pc = simulate_path_counts(h, 10000)
        # FIXME pretty weak test, just establishes that the function works,
        # but not that the value is correct
        @test path_loglikelihood(h, pc) < 0
    end

end

@testset "Log likelihood." begin

    for i in 1:100
        h = random_HMC(3, 2, 5)
        pp = path_probabilities(h)

        for i in 1:100
            path = simulate_path(h)
            @test log(pp[path]) ≈ path_loglikelihood(h, path)
        end
    end

end
